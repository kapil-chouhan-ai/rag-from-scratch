class Generate:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, query, context):
        prompt = f"""
                context :  {context}
                

                query :{query}?
                    """

        messages = f"""
        <|im_start|>system
        You are Dolphin, a helpful AI assistant, \nif answer is not in context reply with 'I don't Know'\n do not use prior knowledge.<|im_end|>
        <|im_start|>user
        {prompt}<|im_end|>
        <|im_start|>assistant
        """

        answer = self.llm(messages, max_new_tokens=300)[0]["generated_text"]
        answer = answer[len(messages) : ]
        return answer