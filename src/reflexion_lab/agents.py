from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord, JudgeResult
from .llm import call_llm, call_llm_json
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    
    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        
        context_str = "\n".join([f"Title: {c.title}\nText: {c.text}" for c in example.context])
        
        for attempt_id in range(1, self.max_attempts + 1):
            # 1. Actor Phase
            actor_user_prompt = f"Question: {example.question}\n\n<Context>\n{context_str}\n</Context>"
            if reflection_memory:
                mem_str = "\n".join([f"- {m}" for m in reflection_memory])
                actor_user_prompt += f"\n\n<ReflectionMemory>\n{mem_str}\n</ReflectionMemory>"
                
            answer, tokens_actor, latency_actor = call_llm(ACTOR_SYSTEM, actor_user_prompt)
            
            # 2. Evaluator Phase
            eval_user_prompt = f"Question: {example.question}\nGold Answer: {example.gold_answer}\nAgent Answer: {answer}"
            judge, tokens_eval, latency_eval = call_llm_json(EVALUATOR_SYSTEM, eval_user_prompt, JudgeResult)
            
            total_tokens = tokens_actor + tokens_eval
            total_latency = latency_actor + latency_eval
            
            final_answer = answer
            final_score = judge.score
            
            if judge.score == 1:
                trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, token_estimate=total_tokens, latency_ms=total_latency)
                traces.append(trace)
                break
                
            # 3. Reflector Phase
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflector_user_prompt = f"""Attempt ID: {attempt_id}
Question: {example.question}
<Context>
{context_str}
</Context>
Agent Answer: {answer}
Gold Answer: {example.gold_answer}
Evaluator Feedback:
Score: {judge.score}
Reason: {judge.reason}
Missing Evidence: {judge.missing_evidence}
Spurious Claims: {judge.spurious_claims}"""

                reflection, tokens_refl, latency_refl = call_llm_json(REFLECTOR_SYSTEM, reflector_user_prompt, ReflectionEntry)
                reflections.append(reflection)
                reflection_memory.append(reflection.next_strategy)
                
                total_tokens += tokens_refl
                total_latency += latency_refl
                
                trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, reflection=reflection, token_estimate=total_tokens, latency_ms=total_latency)
            else:
                trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, token_estimate=total_tokens, latency_ms=total_latency)
                
            traces.append(trace)

        total_tokens_run = sum(t.token_estimate for t in traces)
        total_latency_run = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else "wrong_final_answer"
        
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens_run, latency_ms=total_latency_run, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
