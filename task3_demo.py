import work_flow
import qlib
from qlib.tests.data import GetData
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
from qlib.contrib.report import analysis_position
from qlib.config import REG_CN


if __name__ == "__main__":
    provider = "./qlib_data/cn_data"
    #初始化
    work_flow.init(provider=provider, region=REG_CN)
    data_handler_config = work_flow.data_handler_config(start_time='2008-01-01', end_time='2020-08-01',
                                                       fit_start_time='2008-01-01', fit_end_time='2014-12-31',
                                                       instruments='csi300')
    model_kwargs = {
        'estimator': 'lasso',
        'alpha': 0.5,
    }

    task_model = work_flow.task_model(model_class='LinearModel', module_path='qlib.contrib.model.linear',
                                     m_kwargs=model_kwargs)
    dataset_segments = work_flow.create_segments(train_start='2008-01-01', train_end='2014-12-31',
                                                valid_start='2015-01-01', valid_end='2016-12-31',
                                                test_start='2017-01-01', test_end='2020-08-01')
    dataset_kwargs = work_flow.create_dataset_kwarg(handler_class='Alpha158', kwargs=data_handler_config,
                                                   segment=dataset_segments)
    task_dataset = work_flow.qlibtask_dataset(dataset_class='DatasetH', module_path='qlib.data.dataset',
                                         d_kwargs=dataset_kwargs)
    qlib_task = work_flow.create_task(task_model, task_dataset)
    model, dataset = work_flow.prepare(qlib_task)



    history = work_flow.train_model(qlib_task, model, dataset)
    analysis = work_flow.create_port_analysis_config(freq='day', model=model, dataset=dataset, benchmark="SH000300")
    work_flow.backtest(analysis, history, dataset)