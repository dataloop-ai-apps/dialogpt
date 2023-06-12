import dtlpy as dl
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


@dl.Package.decorators.module(name="model-adapter",
                              description="Model adapter for DialoGPT-large",
                              init_inputs={'model_entity': dl.Model})
class Adapter(dl.BaseModelAdapter):
    tokenizer = None
    model = None

    def load(self, local_path: str, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

    def prepare_item_func(self, item: dl.Item):
        buffer = json.load(item.download(save_locally=False))
        return buffer

    def train(self, data_path, output_path, **kwargs):
        print("Training not implemented yet")

    def predict(self, batch, **kwargs):
        annotations = []
        for item in batch:
            prompts = item["prompts"]
            item_annotations = []
            for prompt_key, prompt_content in prompts.items():
                chat_history_ids = torch.tensor([])
                for question in prompt_content.values():
                    print(f"User: {question['value']}")
                    new_user_input_ids = self.tokenizer.encode(question["value"] + self.tokenizer.eos_token,
                                                               return_tensors='pt')
                    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) \
                        if len(chat_history_ids) else new_user_input_ids
                    chat_history_ids = self.model.generate(bot_input_ids, max_length=1000,
                                                           pad_token_id=self.tokenizer.eos_token_id)
                    response = self.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0],
                                                     skip_special_tokens=True)
                    print("Response: {}".format(response))
                    item_annotations.append({
                        "type": "text",
                        "label": "q",
                        "coordinates": response,
                        "metadata": {"system": {"promptId": prompt_key}}
                    })
            annotations.append(item_annotations)
            return annotations


def package_creation(project: dl.Project):
        metadata = dl.Package.get_ml_metadata(cls=Adapter,
                                              default_configuration={'weights_filename': 'dialogpt.pt',
                                                                     'epochs': 10,
                                                                     'batch': 4,
                                                                     'device': 'cuda:0'},
                                              )
        modules = dl.PackageModule.from_entry_point(entry_point='main.py')

        package = project.packages.push(package_name='dialogpt-runner',
                                        src_path=os.getcwd(),
                                        is_global=False,
                                        package_type='ml',
                                        modules=[modules],
                                        service_config={
                                            'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_GPU_K80_S,
                                                                            autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                                min_replicas=0,
                                                                                max_replicas=1),
                                                                            preemptible=False,
                                                                            concurrency=1).to_json(),
                                            'initParams': {'model_entity': None}
                                        },
                                        ignore_sanity_check=True,
                                        metadata=metadata)
        return package


def model_creation(package: dl.Package):

    model = package.models.create(model_name='pretrained-dialogpt',
                                  description='dialogpt for chatting',
                                  tags=['llm', 'pretrained'],
                                  dataset_id=None,
                                  status='trained',
                                  scope='project',
                                  configuration={
                                      'weights_filename': 'dialogpt.pt',
                                      'device': 'cuda:0'},
                                  project_id=package.project.id
                                  )
    return model


def deploy():
    dl.setenv('rc')
    project_name = '<PROJECT-NAME>'
    project = dl.projects.get(project_name)
    package = package_creation(project=project)
    print(f'new mode pushed. codebase: {package.codebase}')
    model = model_creation(package=package)
    # model_entity = dl.models.get(model_id='640ee84307a569363353ed6a')
    # print(f'model and package deployed. package id: {package.id}, model id: {model_entity.id}')
