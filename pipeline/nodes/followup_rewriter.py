from core.prompt_registry import prompt_registry
from llm.openai_client import OpenAIClient
from schema.followup_rewriter_schema import FollowUpReWriterSchema

llm = OpenAIClient().get_llm()

prompt_registry.load_prompts()


def followup_rewriter(state):

    if state.followup.is_followup != 'true':
        state.resolved_query = state.current_user_query
        return state

    prompt = prompt_registry.get("followup_rewriter").invoke(input={
        "last_result_summary": state.last_result_summary,
        "current_query": state.current_user_query
    })

    response = llm.with_structured_output(FollowUpReWriterSchema).invoke(prompt)
    print(response)

    state.resolved_query = response.rewritten_query.strip()
    return state



if __name__ == '__main__':
    
    from core.state import AgentState
    from langchain_core.messages import HumanMessage
    from schema.follow_up_schema import FollowUpSchema
    
    sample_state = AgentState(
        messages= [HumanMessage(content = "Out of these products which comes under Electronics Category")],
        followup= FollowUpSchema(
            is_followup='True',
            type='new',
            instruction="something here"
        ),
        last_result_summary= "Question: What are the top 10 products by total units sold of all time? Objective: Join order_items with product_variants and products, group by p.name, compute SUM(oi.quantity) as total_units_sold, return p.name and total_units_sold ordered by total_units_sold DESC LIMIT 10 Result: Returned 10 rows with columns: product_name, total_units_sold. Answered: ### Top 10 Products by Total Units Sold of All Time\n\nHere are the leading products based on total units sold:\n\n1. **Adidas Ultraboost 22** - 65 units sold  \n2. **Nike Running Shorts** - 50 units sold  \n3. **Sony Alpha A6400 Camera** - 47 units sold  \n4. **Mamaearth Onion Shampoo** - 45 units sold  \n5. **Ikea LACK Coffee Table** - 41 units sold  \n6. **Mamaearth Vitamin C Serum** - 38 units sold  \n7. **Sony Xperia 5 V** - 36 units sold  \n8. **Philips Electric Kettle** - 36 units sold  \n9. **Wildcraft Backpack 45L** - 35 units sold  \n10. **Levi's 511 Slim Fit Jeans** - 32 units sold  \n\nThese products represent the highest sales figures, indicating strong consumer preference and market demand.",
        current_user_query= "Out of these products which comes under Electronics Category"
        )
    
    
    followup_rewriter(sample_state)
