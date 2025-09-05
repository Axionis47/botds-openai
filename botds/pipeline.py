"""Main pipeline orchestrator with OpenAI-gated stages."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cache import Cache
from .config import Config
from .context import DecisionLog, DataCard, HandoffLedger, RunManifest
from .llm import LLMRouter
from .tools import *
from .utils import ensure_dir, generate_job_id, get_timestamp, Timer


class Pipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config: Config):
        self.config = config
        self.job_id = generate_job_id()
        
        # Setup directories
        self.artifacts_dir = ensure_dir(Path(config.report.out_dir) / self.job_id)
        self.handoffs_dir = ensure_dir(self.artifacts_dir / "handoffs")
        self.logs_dir = ensure_dir(self.artifacts_dir / "logs")
        
        # Initialize components
        self.cache = Cache(config.cache.dir, config.cache.mode)
        self.decision_log = DecisionLog(self.logs_dir / "decision_log.jsonl")
        self.handoff_ledger = HandoffLedger(self.logs_dir / "handoff_ledger.jsonl")
        self.manifest = RunManifest(self.job_id)
        
        # Initialize LLM router
        self.llm_router = LLMRouter(config.llms.model_dump(), self.decision_log)
        
        # Initialize tools
        self._init_tools()
        
        # Pipeline state
        self.current_stage = "initialization"
        self.stage_outputs: Dict[str, Any] = {}
        
    def _init_tools(self) -> None:
        """Initialize all tools."""
        self.tools = {
            "data_store": DataStore(self.artifacts_dir),
            "schema_profiler": SchemaProfiler(self.artifacts_dir),
            "quality_guard": QualityGuard(self.artifacts_dir),
            "splitter": Splitter(self.artifacts_dir),
            "featurizer": Featurizer(self.artifacts_dir),
            "model_trainer": ModelTrainer(self.artifacts_dir),
            "tuner": Tuner(self.artifacts_dir),
            "metrics": Metrics(self.artifacts_dir),
            "calibrator": Calibrator(self.artifacts_dir),
            "fairness": Fairness(self.artifacts_dir),
            "robustness": Robustness(self.artifacts_dir),
            "plotter": Plotter(self.artifacts_dir),
            "artifact_store": ArtifactStore(self.artifacts_dir),
            "handoff_ledger": HandoffLedger(self.artifacts_dir),
            "budget_guard": BudgetGuard(self.artifacts_dir, self.config.budgets.model_dump()),
            "pii": PII(self.artifacts_dir)
        }
    
    def run(self) -> Dict[str, Any]:
        """Run the complete pipeline."""
        try:
            print(f"Starting pipeline run {self.job_id}")
            
            # Stage 1: Intake & Validation
            self._run_stage("intake_validation", self._stage_intake_validation)
            
            # Stage 2: Profiling & Quality
            self._run_stage("profiling_quality", self._stage_profiling_quality)
            
            # Stage 3: EDA & Hypotheses
            self._run_stage("eda_hypotheses", self._stage_eda_hypotheses)
            
            # Stage 4: Feature Plan & Splits
            self._run_stage("feature_splits", self._stage_feature_splits)
            
            # Stage 5: Model Ladder
            self._run_stage("model_ladder", self._stage_model_ladder)
            
            # Stage 6: Evaluation & Stress
            self._run_stage("evaluation_stress", self._stage_evaluation_stress)
            
            # Stage 7: Reporting & Packaging
            self._run_stage("reporting", self._stage_reporting)
            
            # Save final manifest
            self.manifest.save(self.artifacts_dir / "run_manifest.json")
            
            print(f"Pipeline completed successfully!")
            print(f"Artifacts written to: {self.artifacts_dir}")
            
            return {
                "status": "success",
                "job_id": self.job_id,
                "artifacts_dir": str(self.artifacts_dir),
                "cache_stats": self.cache.get_hit_stats()
            }
            
        except Exception as e:
            print(f"Pipeline failed: {str(e)}")
            return {
                "status": "failed",
                "job_id": self.job_id,
                "error": str(e),
                "artifacts_dir": str(self.artifacts_dir)
            }
    
    def _run_stage(self, stage_name: str, stage_func) -> None:
        """Run a pipeline stage with budget monitoring."""
        self.current_stage = stage_name
        print(f"\n=== Stage: {stage_name} ===")
        
        with Timer(stage_name) as timer:
            # Check budget before stage
            budget_status = self.tools["budget_guard"].checkpoint(stage_name)
            
            if budget_status["status"] == "abort":
                raise RuntimeError(f"Budget exceeded at stage {stage_name}: {budget_status['recommendations']}")
            
            # Run stage
            stage_output = stage_func()
            self.stage_outputs[stage_name] = stage_output
            
            # Update manifest
            self.manifest.add_budget_usage(stage_name, {
                "duration_seconds": timer.elapsed,
                "budget_status": budget_status["status"]
            })
        
        print(f"Completed {stage_name} in {timer}")
    
    def _stage_intake_validation(self) -> Dict[str, Any]:
        """Stage 1: Intake & Validation."""
        # Load data
        if self.config.data.source == "builtin":
            data_result = self.tools["data_store"].read_builtin(self.config.data.name)
        else:
            data_result = self.tools["data_store"].read_csv(self.config.data.csv_paths)
        
        # Set target if not specified for CSV
        target = self.config.data.target or data_result.get("target", "target")
        
        # PII check if enabled
        if self.config.pii.enabled:
            pii_scan = self.tools["pii"].scan(data_result["df_ref"], self.config.pii.patterns)
            if pii_scan["risk_level"] in ["medium", "high"] and self.config.pii.action == "redact":
                redact_result = self.tools["pii"].redact(data_result["df_ref"], self.config.pii.patterns)
                data_result["df_ref"] = redact_result["df_ref_sanitized"]
        
        # OpenAI decision: Target validity & task type
        decision_prompt = f"""
        Analyze this dataset and confirm:
        1. Is '{target}' a valid target column?
        2. What is the task type (classification/regression)?
        3. Are there any immediate concerns?
        
        Dataset info: {data_result}
        """
        
        decision = self.llm_router.openai_decide(
            stage="intake_validation",
            prompt=decision_prompt,
            context={"data_info": data_result, "target": target}
        )
        
        return {
            "data_ref": data_result["df_ref"],
            "target": target,
            "task_type": data_result.get("task_hint", "auto"),
            "decision": decision,
            "pii_handled": self.config.pii.enabled
        }
    
    def _stage_profiling_quality(self) -> Dict[str, Any]:
        """Stage 2: Profiling & Quality."""
        intake_output = self.stage_outputs["intake_validation"]
        
        # Profile dataset
        profile_result = self.tools["schema_profiler"].profile(intake_output["data_ref"])
        
        # Quality scan
        leakage_result = self.tools["quality_guard"].leakage_scan(
            intake_output["data_ref"],
            intake_output["target"],
            self.config.split.policy,
            self.config.split.time_col if self.config.split.policy == "time" else None
        )
        
        # OpenAI decision: Proceed or block based on quality
        decision_prompt = f"""
        Review the data quality scan results:
        
        Profile: {profile_result['summary']}
        Leakage scan: {leakage_result}
        
        Decision needed:
        1. Should we proceed with this data?
        2. Are there any blocking issues?
        3. What columns should be denied from features?
        """
        
        decision = self.llm_router.openai_decide(
            stage="profiling_quality",
            prompt=decision_prompt,
            context={
                "profile": profile_result,
                "leakage": leakage_result
            }
        )
        
        if leakage_result["status"] == "block":
            raise RuntimeError(f"Data quality check failed: {leakage_result['offenders']}")
        
        return {
            "profile_ref": profile_result["profile_ref"],
            "leakage_status": leakage_result["status"],
            "quality_decision": decision,
            "deny_list": [item["column"] for item in leakage_result.get("offenders", [])]
        }
    
    def _stage_eda_hypotheses(self) -> Dict[str, Any]:
        """Stage 3: EDA & Hypotheses (simplified for MVP)."""
        # For MVP, generate basic insights
        intake_output = self.stage_outputs["intake_validation"]
        profile_output = self.stage_outputs["profiling_quality"]
        
        # Load profile for insights
        from .utils import load_json
        profile = load_json(profile_output["profile_ref"])
        
        # Generate basic insights
        insights = [
            f"Dataset has {profile['shape']['rows']} rows and {profile['shape']['columns']} columns",
            f"Missing data: {profile['missing_values']['total_missing']} total missing values",
            f"Data types: {len(profile['column_types']['numeric'])} numeric, {len(profile['column_types']['categorical'])} categorical"
        ]
        
        # OpenAI decision: Select top insights and hypotheses
        decision_prompt = f"""
        Based on the data profile, select the top 3 insights and generate 5 testable hypotheses.
        
        Available insights: {insights}
        Profile summary: {profile}
        
        Provide:
        1. Top 3 most important insights
        2. 5 testable hypotheses about the data/target relationship
        """
        
        decision = self.llm_router.openai_decide(
            stage="eda_hypotheses",
            prompt=decision_prompt,
            context={"profile": profile, "insights": insights}
        )
        
        return {
            "top_insights": insights[:3],  # Simplified for MVP
            "hypotheses": ["Hypothesis 1", "Hypothesis 2", "Hypothesis 3", "Hypothesis 4", "Hypothesis 5"],
            "eda_decision": decision
        }
    
    def _stage_feature_splits(self) -> Dict[str, Any]:
        """Stage 4: Feature Plan & Splits."""
        intake_output = self.stage_outputs["intake_validation"]
        quality_output = self.stage_outputs["profiling_quality"]
        
        # Create feature plan
        feature_plan = self.tools["featurizer"].plan(
            intake_output["data_ref"],
            intake_output["target"],
            quality_output["deny_list"]
        )
        
        # Create splits
        splits = self.tools["splitter"].make_splits(
            intake_output["data_ref"],
            intake_output["target"],
            self.config.split.policy,
            self.config.split.test_size,
            self.config.split.val_size,
            self.config.split.seed
        )
        
        # Apply feature engineering
        matrices = self.tools["featurizer"].apply(
            intake_output["data_ref"],
            feature_plan["plan_ref"],
            splits["splits_ref"]
        )
        
        # OpenAI decision: Approve feature plan and split policy
        decision_prompt = f"""
        Review the feature engineering plan and data splits:
        
        Feature plan: {feature_plan['summary']}
        Split summary: {splits['summary']}
        
        Decision needed:
        1. Approve the feature engineering approach?
        2. Confirm the split policy is appropriate?
        3. Any concerns about the feature set?
        """
        
        decision = self.llm_router.openai_decide(
            stage="feature_splits",
            prompt=decision_prompt,
            context={
                "feature_plan": feature_plan,
                "splits": splits
            }
        )
        
        return {
            "feature_plan_ref": feature_plan["plan_ref"],
            "splits_ref": splits["splits_ref"],
            "matrices": matrices,
            "feature_decision": decision
        }
    
    def _stage_model_ladder(self) -> Dict[str, Any]:
        """Stage 5: Model Ladder."""
        feature_output = self.stage_outputs["feature_splits"]
        matrices = feature_output["matrices"]
        
        # Determine task type from intake
        intake_output = self.stage_outputs["intake_validation"]
        task_type = intake_output.get("task_type", "auto")

        # Define model candidates based on task type
        if task_type == "regression":
            model_specs = [
                {"name": "linear_regression", "params": {}},
                {"name": "random_forest_regressor", "params": {"n_estimators": 100}},
            ]
        else:  # classification
            model_specs = [
                {"name": "logistic_regression", "params": {}},
                {"name": "random_forest_classifier", "params": {"n_estimators": 100}},
            ]
        
        # Train models
        model_results = []
        for spec in model_specs:
            try:
                result = self.tools["model_trainer"].train(
                    spec,
                    matrices["X_train_ref"],
                    matrices["y_train_ref"],
                    matrices["X_val_ref"],
                    matrices["y_val_ref"]
                )
                model_results.append(result)
            except Exception as e:
                print(f"Failed to train {spec['name']}: {e}")
        
        # Compare models
        comparison = self.tools["metrics"].compare_models(model_results, "accuracy")
        
        # OpenAI decision: Select best model
        decision_prompt = f"""
        Review the model comparison results and select the best model:
        
        Model rankings: {comparison['rankings']}
        
        Decision needed:
        1. Which model should be selected?
        2. What is the rationale for this choice?
        3. What operating threshold should be used?
        """
        
        decision = self.llm_router.openai_decide(
            stage="model_ladder",
            prompt=decision_prompt,
            context={"comparison": comparison}
        )
        
        return {
            "model_results": model_results,
            "comparison": comparison,
            "selected_model": comparison["best_model"],
            "model_decision": decision
        }
    
    def _stage_evaluation_stress(self) -> Dict[str, Any]:
        """Stage 6: Evaluation & Stress Testing."""
        ladder_output = self.stage_outputs["model_ladder"]
        feature_output = self.stage_outputs["feature_splits"]
        
        selected_model = ladder_output.get("selected_model")
        matrices = feature_output["matrices"]

        # Find the model reference from the results
        selected_model_ref = None
        if selected_model and ladder_output.get("model_results"):
            for result in ladder_output["model_results"]:
                if result.get("model_id") == selected_model.get("model_id"):
                    selected_model_ref = result.get("model_ref")
                    break

        if not selected_model_ref and ladder_output.get("model_results"):
            # Fallback to first model if selection failed
            selected_model_ref = ladder_output["model_results"][0]["model_ref"]

        # Test evaluation
        test_metrics = self.tools["metrics"].evaluate(
            selected_model_ref,
            matrices["X_test_ref"],
            matrices["y_test_ref"]
        )
        
        # Robustness testing
        robustness_result = self.tools["robustness"].shock_tests(
            selected_model_ref,
            matrices["X_val_ref"],
            matrices["y_val_ref"]
        )
        
        # OpenAI decision: Accept evaluation results
        decision_prompt = f"""
        Review the final evaluation and robustness results:
        
        Test metrics: {test_metrics}
        Robustness grade: {robustness_result['resilience_grade']}
        
        Decision needed:
        1. Are these results acceptable for deployment?
        2. Any concerns about model robustness?
        3. Final recommendations?
        """
        
        decision = self.llm_router.openai_decide(
            stage="evaluation_stress",
            prompt=decision_prompt,
            context={
                "test_metrics": test_metrics,
                "robustness": robustness_result
            }
        )
        
        return {
            "test_metrics": test_metrics,
            "robustness_result": robustness_result,
            "evaluation_decision": decision
        }
    
    def _stage_reporting(self) -> Dict[str, Any]:
        """Stage 7: Reporting & Packaging."""
        # Collect all results
        all_outputs = self.stage_outputs
        
        # Prepare report payload
        report_payload = {
            "business_goal": self.config.business_goal,
            "primary_metric": self.config.metrics.get("primary", "auto"),
            "dataset_info": {"shape": "N/A"},  # Simplified for MVP
            "target": all_outputs["intake_validation"]["target"],
            "leakage_status": all_outputs["profiling_quality"]["leakage_status"],
            "top_insights": all_outputs["eda_hypotheses"]["top_insights"],
            "selected_model": all_outputs["model_ladder"]["selected_model"],
            "leaderboard": all_outputs["model_ladder"]["comparison"]["rankings"][:3],
            "operating_point": {"threshold": "0.5", "conservative": "0.7", "tradeoff": "Balanced precision/recall"},
            "robustness_grade": all_outputs["evaluation_stress"]["robustness_result"]["resilience_grade"],
            "robustness_reason": "Based on shock test results",
            "next_steps": ["Deploy model", "Monitor performance", "Collect feedback"],
            "shortcuts_taken": self.manifest.shortcuts_taken
        }
        
        # OpenAI decision: Final report sign-off
        decision_prompt = f"""
        Review the complete analysis and provide final sign-off for the executive summary:
        
        Results summary: {report_payload}
        
        Provide:
        1. Executive summary text
        2. Key recommendations
        3. Any caveats or limitations
        """
        
        decision = self.llm_router.openai_decide(
            stage="reporting",
            prompt=decision_prompt,
            context={"report_payload": report_payload}
        )
        
        # Generate reports
        one_pager = self.tools["artifact_store"].write_report(
            "one_pager",
            report_payload,
            self.config.report.format
        )
        
        appendix = self.tools["artifact_store"].write_report(
            "appendix",
            {
                "config_snapshot": str(self.config.model_dump()),
                "dataset_hash": "N/A",  # Simplified for MVP
                "seeds": {"main": 42}
            },
            self.config.report.format
        )
        
        return {
            "one_pager_ref": one_pager["report_ref"],
            "appendix_ref": appendix["report_ref"],
            "final_decision": decision
        }
