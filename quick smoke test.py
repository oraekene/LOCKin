# quick manual test
from persistence_helpers import save_user_backtest_config, list_user_backtest_configs, load_user_backtest_config, append_backtest_history, save_forwarding_job, load_forwarding_job

uid = 12345
cfg = {"name":"test","market_filters":{"min_market_cap":1000000}}
cid = save_user_backtest_config(uid, "test cfg", cfg)
print("cfg id", cid)
print("saved list:", list_user_backtest_configs(uid))
print("loaded cfg:", load_user_backtest_config(uid, cid))
jid = save_forwarding_job(uid, {"name":"fwd1","filters":{}, "cooldown_seconds_per_asset":3600})
print("job id", jid)
print("loaded job", load_forwarding_job(uid, jid))
runid = append_backtest_history(uid, {"name":"run1","metadata":{"pnl":12.3}})
print("run id", runid)
print("history page", list_backtest_history(uid, page=1))
