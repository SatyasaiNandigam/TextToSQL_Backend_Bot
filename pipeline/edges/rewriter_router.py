REUSE_TYPES = {"explain", "visualize"}
REWRITE_TYPES = {"refine", "regroup", "drilldown"}


def route_after_rewriter(state):
    followup = state.followup
    if followup and followup.is_followup == 'True':
        if followup.type in REUSE_TYPES:
            return "reuse"
        if followup.type in REWRITE_TYPES:
            return "rewrite"
    return "fresh"
