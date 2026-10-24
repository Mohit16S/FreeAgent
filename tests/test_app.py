"""Exercise the app using actual workspace data if present; no synthetic UI output."""
from streamlit.testing.v1 import AppTest
from pathlib import Path

def test_overview_loads():
    app=AppTest.from_file(str(Path('app.py').resolve())).run(timeout=30)
    assert not app.exception
    assert 'Buying Points' in app.title[0].value


def test_pages_handle_untrained_state():
    app=AppTest.from_file(str(Path('app.py').resolve())).run(timeout=30)
    for page in ['Transfer evaluator','Player impact','Counterfactual analysis','Net Value','Club objectives','Backtesting','Explainability']:
        app.sidebar.radio[0].set_value(page).run(timeout=30)
        assert not app.exception


def test_real_scenario_form_and_result_pages():
    import pytest
    if not Path('models/artifacts/core.joblib').exists(): pytest.skip('Real-data artifact unavailable; never substitute synthetic UI data')
    app=AppTest.from_file(str(Path('app.py').resolve())).run(timeout=30)
    app.sidebar.radio[0].set_value('Transfer evaluator').run(timeout=30)
    app.button[0].click().run(timeout=30)
    assert app.error
    app.selectbox[1].set_value('24').run(timeout=30)
    app.selectbox[2].set_value('3535')
    app.selectbox[3].set_value('10955')
    for widget in app.number_input:
        if widget.label=='Assumed price per league point (€)': widget.set_value(2000000.)
    app.text_input[0].set_value('Software acceptance test with explicit hypothetical cost assumptions')
    app.button[0].click().run(timeout=60)
    assert not app.exception and not app.error
    assert 'result' in app.session_state
    for page in ['Counterfactual analysis','Net Value','Club objectives','Explainability']:
        app.sidebar.radio[0].set_value(page).run(timeout=30)
        assert not app.exception
