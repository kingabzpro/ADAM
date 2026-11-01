from nicegui import ui


def dashboard() -> None:
    ui.label("ADAM").classes("text-h4")
    ui.label("Local market research and decision support")
    ui.label("Analysis is not available yet.")


def run_dashboard() -> None:
    ui.run(dashboard, title="ADAM", host="127.0.0.1", reload=False)
