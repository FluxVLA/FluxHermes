"""Execute the deployment transform chain declared in configuration."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from .apis import import_config_plugins
from .contracts import DeploymentConfig
from .core import PipelineManager
from .plugins import add_python_paths, load_symbol
from .transforms import Compose
from .utils import write_json


class CompilePipeline:
    """Build a config-defined transform chain with timing and failure reports."""

    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config
        self.pipeline_manager = PipelineManager()
        import_config_plugins(config)
        add_python_paths([
            *config.source.get('python_paths', ()),
            *config.codebase.get('python_paths', ()),
        ])
        for hook in config.pipeline_hooks:
            self.pipeline_manager.add_hook(
                stage=hook['stage'],
                hook=load_symbol(hook['hook']),
                when=hook.get('when', 'after'))
        self.transforms = Compose(config.pipeline)
        self.context = self.new_context('')
        self.transforms.validate(set(self.context), config)

    def new_context(self, run_id):
        return dict(
            config=self.config,
            manager=self.pipeline_manager,
            run_id=run_id,
            device=str(
                self.config.codebase.get('device',
                                         self.config.source.get('device', 'cuda:0'))),
            output=self.config.work_dir,
            validation=None)

    def plan(self):
        return self.transforms.plan()

    def plan_dict(self):
        config = self.config
        return dict(
            name=config.name,
            work_dir=str(config.work_dir),
            codebase=config.codebase.get('type'),
            task=config.codebase.get('task'),
            ir=config.ir['type'],
            backend=config.backend['type'],
            precision=config.precision,
            stages=self.plan(),
            pipeline=config.pipeline,
            inputs=[item.to_dict() for item in config.inputs],
            outputs=[item.to_dict() for item in config.outputs])

    def execute(self) -> Path:
        self.context = self.new_context(uuid4().hex)
        self.pipeline_manager.records.clear()
        work_dir = self.config.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        write_json(work_dir / 'resolved_config.json', self.config.to_dict())
        write_json(
            work_dir / 'result.json',
            dict(
                run_id=self.context['run_id'],
                status='running',
                workflow_completed=False,
                numerical_passed=None))
        try:
            self.context = self.transforms(self.context)
        except Exception as error:
            validation = getattr(error, 'report', self.context.get('validation'))
            report = dict(
                run_id=self.context['run_id'],
                status='failed',
                workflow_completed=False,
                numerical_passed=validation.get('passed') if isinstance(
                    validation, dict) else None,
                error=dict(type=type(error).__name__, message=str(error)),
                validation=validation,
                stages=self.pipeline_manager.report())
            write_json(work_dir / 'failure.json', report)
            write_json(work_dir / 'result.json', report)
            raise
        validation = self.context.get('validation')
        write_json(
            work_dir / 'result.json',
            dict(
                run_id=self.context['run_id'],
                status='completed',
                workflow_completed=True,
                numerical_passed=validation.get('passed') if validation else None,
                output=str(self.context['output']),
                pipeline=self.pipeline_manager.report()))
        return self.context['output']
