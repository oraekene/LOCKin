import os
import json
import shutil
import pytest

def _set_data_dir(tmp_path):
    test_dir = str(tmp_path / "data")
    os.environ["DATA_DIR"] = test_dir
    return test_dir

def test_save_list_load_delete_backtest_configs_and_history(tmp_path):
    test_dir = _set_data_dir(tmp_path)

    import persistence_helpers as ph

    user_id = 4242

    # --- save a backtest config ---
    cfg = {
        "market_filters": {"min_market_cap": 1000000},
        "strategy": {"name": "quick-test", "params": {}}
    }
    cfg_name = "cfg-unit-test"
    cfg_id = ph.save_user_backtest_config(user_id, cfg_name, cfg)
    assert cfg_id, "save_user_backtest_config should return an id"

    # list configs
    configs = ph.list_user_backtest_configs(user_id)
    assert isinstance(configs, list)
    assert any(c.get("id") == cfg_id for c in configs), "Saved config missing from list"

    # load config
    loaded_cfg = ph.load_user_backtest_config(user_id, cfg_id)
    assert isinstance(loaded_cfg, dict)
    assert loaded_cfg.get("market_filters", {}).get("min_market_cap") == 1000000

    # --- append backtest history ---
    run_meta = {"name": "run1", "metadata": {"pnl": 12.5}}
    run_id = ph.append_backtest_history(user_id, run_meta)
    assert run_id, "append_backtest_history should return run id"

    # list history
    hist_page = ph.list_backtest_history(user_id, page=1, per_page=5)
    assert isinstance(hist_page, dict)
    assert hist_page["total"] >= 1
    assert any(item.get("id") == run_id for item in hist_page["items"])

    # verify result_path written (if any)
    # load run metadata via reading user obj
    user_obj = ph.read_user_file(user_id)
    history = user_obj.get("backtest_history", [])
    assert any(h.get("id") == run_id for h in history)

    # --- delete config using delete_user_backtest_config ---
    deleted = ph.delete_user_backtest_config(user_id, cfg_id)
    # delete may return True if deleted or False if file missing; assert that config is gone
    new_configs = ph.list_user_backtest_configs(user_id)
    assert not any(c.get("id") == cfg_id for c in new_configs), "Config still present after delete_user_backtest_config"

    # cleanup
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir, ignore_errors=True)
