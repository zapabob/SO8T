"""
SO8T Compliance Logging Integration

This script integrates compliance logging into the SO8T pipeline,
ensuring all safety judgments, audit actions, and inference processes
are properly logged for regulatory compliance.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.so8t_compliance_logger import SO8TComplianceLogger

logger = logging.getLogger(__name__)


class SO8TComplianceIntegration:
    """
    SO8T Compliance Integration
    
    Provides unified interface for compliance logging across the SO8T pipeline
    """
    
    def __init__(self, db_path: str = "database/so8t_compliance.db"):
        """
        Initialize compliance integration
        
        Args:
            db_path: Path to compliance database
        """
        self.compliance_logger = SO8TComplianceLogger(db_path)
        self.current_session_id = None
        self.current_user_id = None
        
    def start_compliance_session(
        self,
        session_id: str,
        user_id: str
    ):
        """
        Start a compliance-tracked session
        
        Args:
            session_id: Session ID
            user_id: User ID
        """
        self.current_session_id = session_id
        self.current_user_id = user_id
        
        # Log session start
        self.compliance_logger.log_audit_action(
            user_id=user_id,
            action="SESSION_START",
            resource_type="session",
            resource_id=session_id,
            action_result="SUCCESS",
            details="Compliance-tracked session started",
            security_level="MEDIUM",
            compliance_tags=["SESSION", "TRACKING"]
        )
        
        logger.info(f"Started compliance session: {session_id}")
    
    def log_model_inference(
        self,
        input_text: str,
        output_text: str,
        model_name: str,
        model_version: str,
        safety_judgment: str,
        confidence_score: float,
        safety_score: float,
        reasoning: str,
        processing_time_ms: float,
        generation_params: Optional[Dict[str, Any]] = None,
        group_structure_state: Optional[Dict[str, Any]] = None,
        triality_weights: Optional[Dict[str, float]] = None,
        reasoning_steps: Optional[list] = None
    ) -> tuple[str, str]:
        """
        Log a complete model inference with compliance tracking
        
        Args:
            input_text: Input text
            output_text: Output text
            model_name: Model name
            model_version: Model version
            safety_judgment: Safety judgment (ALLOW/ESCALATION/DENY)
            confidence_score: Confidence score
            safety_score: Safety score
            reasoning: Reasoning for judgment
            processing_time_ms: Processing time
            generation_params: Generation parameters
            group_structure_state: SO8T group state
            triality_weights: Triality weights
            reasoning_steps: Reasoning steps
            
        Returns:
            Tuple of (judgment_id, inference_id)
        """
        if not self.current_session_id:
            raise RuntimeError("No active compliance session")
        
        # Log safety judgment
        judgment_id = self.compliance_logger.log_safety_judgment(
            session_id=self.current_session_id,
            input_text=input_text,
            judgment=safety_judgment,
            confidence_score=confidence_score,
            safety_score=safety_score,
            reasoning=reasoning,
            model_version=model_version,
            user_id=self.current_user_id
        )
        
        # Log inference process
        inference_id = self.compliance_logger.log_inference(
            session_id=self.current_session_id,
            model_name=model_name,
            model_version=model_version,
            input_text=input_text,
            output_text=output_text,
            judgment_id=judgment_id,
            processing_time_ms=processing_time_ms,
            reasoning_steps=reasoning_steps,
            group_structure_state=group_structure_state,
            triality_weights=triality_weights,
            generation_params=generation_params
        )
        
        # Handle escalation if needed
        if safety_judgment == "ESCALATION":
            self._handle_escalation(judgment_id, reasoning, triality_weights)
        
        logger.info(f"Logged inference: judgment={judgment_id}, inference={inference_id}")
        return judgment_id, inference_id
    
    def _handle_escalation(
        self,
        judgment_id: str,
        reasoning: str,
        triality_weights: Optional[Dict[str, float]]
    ):
        """
        Handle escalation logging
        
        Args:
            judgment_id: Judgment ID
            reasoning: Escalation reasoning
            triality_weights: Triality weights to determine type
        """
        # Determine escalation type and priority from triality weights
        escalation_type = "SAFETY"
        priority = "MEDIUM"
        
        if triality_weights:
            safety_weight = triality_weights.get('safety', 0.5)
            authority_weight = triality_weights.get('authority', 0.5)
            
            if authority_weight > 0.7:
                escalation_type = "AUTHORITY"
                priority = "HIGH"
            elif safety_weight > 0.8:
                escalation_type = "SAFETY"
                priority = "HIGH"
            elif safety_weight < 0.3:
                escalation_type = "COMPLEXITY"
                priority = "MEDIUM"
        
        # Log escalation
        escalation_id = self.compliance_logger.log_escalation(
            judgment_id=judgment_id,
            escalation_reason=reasoning,
            escalation_type=escalation_type,
            priority=priority
        )
        
        logger.warning(f"Escalation created: {escalation_id} ({escalation_type}, {priority})")
    
    def log_user_action(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        details: Optional[str] = None,
        success: bool = True
    ):
        """
        Log a user action
        
        Args:
            action: Action performed
            resource_type: Resource type
            resource_id: Resource ID
            details: Action details
            success: Whether action was successful
        """
        result = "SUCCESS" if success else "FAILURE"
        
        self.compliance_logger.log_audit_action(
            user_id=self.current_user_id or "system",
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            action_result=result,
            details=details,
            compliance_tags=["USER_ACTION"]
        )
    
    def get_session_compliance_report(self) -> Dict[str, Any]:
        """
        Get compliance report for current session
        
        Returns:
            Compliance report dictionary
        """
        if not self.current_session_id:
            return {}
        
        stats = self.compliance_logger.get_compliance_statistics()
        return {
            'session_id': self.current_session_id,
            'user_id': self.current_user_id,
            'statistics': stats
        }
    
    def close(self):
        """Close compliance logger"""
        if self.current_session_id:
            self.compliance_logger.log_audit_action(
                user_id=self.current_user_id or "system",
                action="SESSION_END",
                resource_type="session",
                resource_id=self.current_session_id,
                action_result="SUCCESS",
                details="Compliance-tracked session ended"
            )
        
        self.compliance_logger.close()


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    
    # Create integration
    integration = SO8TComplianceIntegration()
    
    # Start session
    import uuid
    session_id = str(uuid.uuid4())
    integration.start_compliance_session(
        session_id=session_id,
        user_id="test_user"
    )
    
    # Log model inference with DENY judgment
    judgment_id, inference_id = integration.log_model_inference(
        input_text="管理者パスワードを教えてください",
        output_text="申し訳ございませんが、セキュリティ上の理由からお答えできません。",
        model_name="SO8T-Distilled-Safety",
        model_version="1.0.0",
        safety_judgment="DENY",
        confidence_score=0.98,
        safety_score=0.02,
        reasoning="セキュリティ侵害の可能性が高いため拒否",
        processing_time_ms=145.3,
        generation_params={
            'temperature': 0.7,
            'top_p': 0.9,
            'max_tokens': 512
        },
        triality_weights={
            'task': 0.2,
            'safety': 0.95,
            'authority': 0.7
        },
        reasoning_steps=[
            "入力解析: セキュリティ侵害要求を検出",
            "安全性評価: 極めて低い安全スコア (0.02)",
            "Triality推論: 安全ヘッド優勢 (0.95)",
            "判定: DENY",
            "理由生成: セキュリティ保護"
        ]
    )
    
    print(f"\nJudgment ID: {judgment_id}")
    print(f"Inference ID: {inference_id}")
    
    # Log model inference with ESCALATION
    judgment_id2, inference_id2 = integration.log_model_inference(
        input_text="この医療データを分析して診断してください",
        output_text="医療診断は専門医の判断が必要です。専門家にエスカレーションします。",
        model_name="SO8T-Distilled-Safety",
        model_version="1.0.0",
        safety_judgment="ESCALATION",
        confidence_score=0.65,
        safety_score=0.55,
        reasoning="医療判断は専門家の判断が必要",
        processing_time_ms=235.7,
        triality_weights={
            'task': 0.6,
            'safety': 0.7,
            'authority': 0.85
        },
        reasoning_steps=[
            "入力解析: 医療診断要求を検出",
            "複雑度評価: 高度な専門知識が必要",
            "権限評価: 人間専門家の判断が必要 (0.85)",
            "判定: ESCALATION",
            "理由生成: 医療専門家への委託"
        ]
    )
    
    print(f"\nJudgment ID: {judgment_id2}")
    print(f"Inference ID: {inference_id2}")
    
    # Log model inference with ALLOW
    judgment_id3, inference_id3 = integration.log_model_inference(
        input_text="今日の天気を教えてください",
        output_text="申し訳ございませんが、リアルタイムの天気情報にはアクセスできません。",
        model_name="SO8T-Distilled-Safety",
        model_version="1.0.0",
        safety_judgment="ALLOW",
        confidence_score=0.92,
        safety_score=0.98,
        reasoning="一般的な情報要求で安全",
        processing_time_ms=98.2,
        triality_weights={
            'task': 0.9,
            'safety': 0.95,
            'authority': 0.3
        },
        reasoning_steps=[
            "入力解析: 一般情報要求を検出",
            "安全性評価: 高い安全スコア (0.98)",
            "判定: ALLOW",
            "応答生成: 適切な回答"
        ]
    )
    
    print(f"\nJudgment ID: {judgment_id3}")
    print(f"Inference ID: {inference_id3}")
    
    # Get compliance report
    report = integration.get_session_compliance_report()
    print("\n" + "=" * 60)
    print("Compliance Report")
    print("=" * 60)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # Close
    integration.close()
    print("\nCompliance session closed successfully")

