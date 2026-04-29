from core.prompt_loader import load_prompt


class PromptRegistry:

    def __init__(self):
        self.prompts = {}

    def load_prompts(self):

        self.prompts["intent_classifier"] = load_prompt("prompts/intent_classifier_prompt.yaml")
        self.prompts["query_planner"] = load_prompt("prompts/query_planner_prompt.yaml")
        self.prompts["sql_agent"] = load_prompt("prompts/sql_agent_prompt.yaml")
        self.prompts["sql_validaton"] = load_prompt("prompts/validation_prompt_v5.yaml")
        self.prompts["followup_detector"] = load_prompt("prompts/follow_up_prompt.yaml")
        self.prompts["nl_response"] = load_prompt("prompts/nl_response_prompt_v3.yaml")
        self.prompts["followup_rewriter"] = load_prompt("prompts/followup_rewriter_prompt.yaml")
        self.prompts["schema_summarizer"] = load_prompt("prompts/schema_summarizer_prompt_v5.yaml")


    def get(self, name: str):
        return self.prompts[name]


prompt_registry = PromptRegistry()