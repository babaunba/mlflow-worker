import asyncio
import json
from datetime import timedelta
import model

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio import workflow
from temporalio import activity

from gen.proto.labels.v1.labels_pb2 import GetLabelsRequest, GetLabelsResponse

loaded_model = None

workflow_name = "get-labels-workflow"
activity_name = "get-labels-activity"
task_queue = "labels-tasks"

@activity.defn(name=activity_name)
async def GetLabelsActivity(issue: model.Issue) -> list[str]:
    global loaded_model

    if loaded_model is None:
        with open('data.json') as file:
            args=json.load(file)
        loaded_model = model.load_model(
            mlflow_server_uri=args['mlflow_server'],
            model_run_id=args['model_run_id'],
            model_name=args['model_name'],
            model_version=args['model_version'],
        )

    return loaded_model.run(issue)

@workflow.defn(name=workflow_name, sandboxed=False)
class GetLabelsWorkflow:
    @workflow.run
    async def run(self, req: GetLabelsRequest) -> GetLabelsResponse:
        issue = model.Issue(
            title=req.title,
            body=req.body,
            labels=list(req.labels),
            creator=req.creator,
        )

        suggested_labels = await workflow.execute_activity(
                GetLabelsActivity,
                issue,
                start_to_close_timeout=timedelta(minutes=10),
        )

        return GetLabelsResponse(labels=suggested_labels)


async def main():
    client = await Client.connect(target_host="localhost:7233")
    worker = Worker(
       client,
       task_queue=task_queue,
       workflows=[GetLabelsWorkflow],
       activities=[GetLabelsActivity],
    )
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
