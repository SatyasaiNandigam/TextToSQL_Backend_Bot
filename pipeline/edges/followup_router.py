def route_after_followup(state):
    if state.followup:
        followup_info = state.followup
        print(followup_info)
        print(followup_info.type)
        print(followup_info.is_followup)
        if followup_info.is_followup == 'true':
            if followup_info.type == "explain":
                return "insight"
       
        return "continue"
    