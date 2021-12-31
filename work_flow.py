import qlib
from qlib.tests.data import GetData
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
from qlib.contrib.report import analysis_position
from qlib.config import REG_CN

#初始化
def init(provider, region):
    qlib.init(provider_uri=provider, region=region, expression_cache=None, dataset_cache=None)

#模型调用以及声明

def task_model(model_class, module_path, m_kwargs):
    model = {
        'class': model_class,
        'module_path': module_path,
        'kwargs': m_kwargs,
    }
    return model

#获取数据
def get_data(provider, region):
    GetData().qlib_data(target_dir=provider, region=region)
#配置
def data_handler_config(start_time, end_time, fit_start_time, fit_end_time, instruments):
    dhc = {
        'start_time': start_time,
        'end_time': end_time,
        'fit_start_time': fit_start_time,
        'fit_end_time': fit_end_time,
        'instruments': str(instruments)
    }
    return dhc

def create_segments(train_start, train_end, valid_start, valid_end, test_start, test_end):
    segment = {
        'train': (train_start, train_end),
        'valid': (valid_start, valid_end),
        'test': (test_start, test_end),
    }
    return segment


def create_dataset_kwarg(handler_class, kwargs, segment):
    dataset_kwargs = {
        'handler': {
            'class': handler_class,
            'module_path': 'qlib.contrib.data.handler',
            'kwargs': kwargs
        },
        'segments': segment
    }
    return dataset_kwargs


def qlibtask_dataset(dataset_class, module_path, d_kwargs):
    dataset = {
        'class': dataset_class,
        'module_path': module_path,
        'kwargs': d_kwargs,
    }
    return dataset


def create_task(model, dataset):
    task = {
        'model': model,
        'dataset': dataset,
    }
    return task


def prepare(task):
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])
    # 需要对数据进行预处理
    return model, dataset


def train_model(task, model, dataset):
    with R.start(experiment_name="train_model"):
        R.log_params(**flatten_dict(task))
        model.fit(dataset)
        R.save_objects(trained_model=model)
        rid = R.get_recorder().id
    return rid


def create_port_analysis_config(freq, model, dataset, benchmark):
    pac = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": freq,
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
                "freq": freq,
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }
    return pac


def backtest(port_analysis_config, rid, dataset):
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
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    report_normal_df = report_normal_df.dropna(axis=0)
    label_df = dataset.prepare("test", col_set="label")
    label_df.columns = ['label']
    analysis_position.report_graph(report_normal_df)

