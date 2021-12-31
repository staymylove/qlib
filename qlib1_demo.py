import qlib
import pandas as pd
from qlib.config import REG_CN
from qlib.contrib.report import analysis_model, analysis_position
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict


if __name__ == "__main__":
    # use default data NOTE: need to download data from remote: python scripts/get_data.py qlib_data_cn --target_dir
    # ~/.qlib/qlib_data/cn_data
    provider_uri = "D:/qlib_data/cn_data"  # target_dir

    qlib.init(provider_uri=provider_uri, region=REG_CN, expression_cache=None, dataset_cache=None)
    market = "csi300"
    benchmark = "SH000300"
    ###################################
    # train model
    ###################################
    data_handler_config = {
        "start_time": "2008-01-01",
        "end_time": "2020-08-01",
        "fit_start_time": "2008-01-01",
        "fit_end_time": "2014-12-31",
        "instruments": market,
    }

    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 20,
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": ("2008-01-01", "2014-12-31"),
                    "valid": ("2015-01-01", "2016-12-31"),
                    "test": ("2017-01-01", "2020-08-01"),
                },
            },
        },
    }

    # model initiaiton
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    # start exp to train model
    with R.start(experiment_name="train_model"):
        R.log_params(**flatten_dict(task))
        model.fit(dataset)
        R.save_objects(trained_model=model)
        rid = R.get_recorder().id

    # prediction, backtest & analysis
    ###################################
    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "model": model,
                "dataset": dataset,
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }
    # backtest and analysis
    with R.start(experiment_name="backtest_analysis"):
        recorder = R.get_recorder(recorder_id=rid, experiment_name="train_model")
        model = recorder.load_object("trained_model")

        # prediction
        recorder = R.get_recorder()
        ba_rid = recorder.id
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # backtest & analysis
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()

    recorder = R.get_recorder(recorder_id=ba_rid, experiment_name="backtest_analysis")
    print(recorder)
    #test
    pred_df = recorder.load_object("pred.pkl")
    pred_df_dates = pred_df.index.get_level_values(level='datetime')
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    analysis_position.report_graph(report_normal_df)
    analysis_position.risk_analysis_graph(analysis_df, report_normal_df)
    label_df = dataset.prepare("test", col_set="label")
    label_df.columns = ['label']
    pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
    analysis_position.score_ic_graph(pred_label)
    analysis_model.model_performance_graph(pred_label)