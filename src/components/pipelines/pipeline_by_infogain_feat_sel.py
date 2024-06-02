from azure.ai.ml import dsl, Input
from .base_pipeline_factory import BasePipelineFactory

class NBCPipelineFactory(BasePipelineFactory):
    @dsl.pipeline(
        name="infogain-feat-sel-nbc-train-pipeline",
        compute="serverless",
        description="E2E data_perp-train pipeline",
    )
    def create_pipeline(self, data_to_train: Input, data_to_test: Input):
        data_prep_job = self.data_prep_component(
            data_to_train=data_to_train,
            data_to_test=data_to_test,
            flag_remove_null_values=False,
            flag_remove_values_by_percentage=False,
            percentage_to_remove_column=0,
        )

        infogain_job = self.infogain_component(
            train_data=data_prep_job.outputs.train_data,
            test_data=data_prep_job.outputs.test_data,
            feature_percentage=0.5,
        )

        train_nbc_job = self.train_nbc(
            train_data=infogain_job.outputs.train_data_feat_sel,  
            test_data=infogain_job.outputs.test_data_feat_sel,   
        )