from pipeline.graph import app
from pathlib import Path

output_path = Path(__file__).parent / "graph_v1.png"

try:
    print(app.get_graph().draw_mermaid())
    with open(output_path, "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())
except Exception as e:
    print("Exception: ",str(e))