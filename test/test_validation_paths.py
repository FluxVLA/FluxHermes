"""Validation manifests resolve beside the deployment, not the launch directory."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fluxhermes.adapters.base import PreparedModel
from fluxhermes.config import load_config
from fluxhermes.contracts import DeploymentConfig
from fluxhermes.validation.parity import validate_backend


def deployment(manifest):
    return dict(
        name='paths',
        work_dir='work',
        source=dict(type='python', factory='unused:factory'),
        inputs=[dict(name='x', dtype='float32', shape=(1, 2))],
        outputs=[dict(name='y', dtype='float32', shape=(1, 2))],
        pipeline=[dict(type='PrepareModel')],
        validation=dict(samples_manifest=manifest),
    )


class IdentityBackend:

    def infer(self, inputs, synchronize=False):
        assert synchronize
        return {'y': inputs['x'].clone()}


@pytest.mark.parametrize('decoy', [False, True])
def test_manifest_and_samples_follow_config_directory(tmp_path, monkeypatch, decoy):
    config_dir = tmp_path / 'configuration'
    config_dir.mkdir()
    samples_dir = config_dir / 'fixtures'
    samples_dir.mkdir()
    np.savez(samples_dir / 'input.npz', x=np.array([[1.25, -2.5]], dtype=np.float32))
    (config_dir / 'samples.json').write_text(
        json.dumps({
            'samples': [{
                'id': 'correct-config-sample',
                'inputs': 'fixtures/input.npz'
            }]
        }),
        encoding='utf-8')
    config_path = config_dir / 'deploy.py'
    config_path.write_text(
        'deployment = ' + repr(deployment('samples.json')), encoding='utf-8')
    launch_dir = tmp_path / 'launch'
    launch_dir.mkdir()
    if decoy:
        np.savez(launch_dir / 'input.npz', x=np.zeros((1, 2), dtype=np.float32))
        (launch_dir / 'samples.json').write_text(
            json.dumps(
                {'samples': [{
                    'id': 'wrong-launch-sample',
                    'inputs': 'input.npz'
                }]}),
            encoding='utf-8')
    monkeypatch.chdir(launch_dir)
    config = load_config(config_path)
    prepared = PreparedModel(torch.nn.Identity(), {'x': torch.ones(1, 2)}, ('y', ))
    report = validate_backend(prepared, IdentityBackend(), config.validation)
    assert report['passed']
    assert report['samples'][0]['id'] == 'correct-config-sample'
    assert Path(config.validation['samples_manifest']) == config_dir / 'samples.json'


def test_from_dict_normalizes_manifest_without_mutating_caller(tmp_path):
    original = deployment('fixtures/samples.json')
    config = DeploymentConfig.from_dict(original, config_path=tmp_path / 'deploy.py')
    assert config.validation['samples_manifest'] == str(tmp_path /
                                                        'fixtures/samples.json')
    assert original['validation']['samples_manifest'] == 'fixtures/samples.json'


def test_manifest_absolute_path_and_environment_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv('SAMPLE_ROOT', str(tmp_path))
    config_path = tmp_path / 'deploy.py'
    config_path.write_text(
        'deployment = ' + repr(deployment('$SAMPLE_ROOT/samples.json')),
        encoding='utf-8')
    config = load_config(config_path)
    assert config.validation['samples_manifest'] == str(tmp_path / 'samples.json')


def test_direct_validation_without_config_keeps_cwd_semantics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    np.savez('input.npz', x=np.ones((1, 2), dtype=np.float32))
    Path('samples.json').write_text(
        json.dumps({'samples': [{
            'id': 'direct-api',
            'inputs': 'input.npz'
        }]}),
        encoding='utf-8')
    prepared = PreparedModel(torch.nn.Identity(), {'x': torch.ones(1, 2)}, ('y', ))
    config = DeploymentConfig.from_dict(deployment('samples.json'))
    report = validate_backend(prepared, IdentityBackend(), config.validation)
    assert report['passed']
    assert report['samples'][0]['id'] == 'direct-api'
