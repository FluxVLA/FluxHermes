import unittest

from fluxhermes.model_families.groot_n15.contract import GR00TN15Contract
from fluxhermes.model_families.base import config_nodes
from fluxhermes.model_families.pi05.contract import PI05Contract


class ModelFamilyContractTest(unittest.TestCase):

    def test_pi05_contract_uses_training_horizon(self):
        contract = PI05Contract(
            image_views=3,
            action_horizon=50,
            denoise_steps=8,
        )

        self.assertEqual(contract.image_views, 3)
        self.assertEqual(contract.action_horizon, 50)
        self.assertEqual(contract.denoise_steps, 8)

    def test_groot_contract_uses_training_horizon_and_output_dimension(self):
        contract = GR00TN15Contract(
            image_views=3,
            sequence_length=900,
            action_horizon=32,
            output_action_dim=14,
        )

        self.assertEqual(contract.image_views, 3)
        self.assertEqual(contract.sequence_length, 900)
        self.assertEqual(contract.action_horizon, 32)
        self.assertEqual(contract.output_action_dim, 14)
        self.assertEqual(contract.sa_length, 65)

    def test_groot_rejects_output_dimension_larger_than_engine_action(self):
        with self.assertRaisesRegex(ValueError, 'cannot exceed'):
            GR00TN15Contract(action_dim=14, output_action_dim=32)

    def test_finds_repeated_typed_training_config_nodes(self):
        config = {
            'train': {
                'transforms': [{
                    'type': 'ProcessPromptsWithImage',
                    'max_len': 900,
                }]
            },
            'eval': {
                'transform': {
                    'type': 'ProcessPromptsWithImage',
                    'max_len': 900,
                }
            },
        }

        nodes = config_nodes(config, 'ProcessPromptsWithImage')

        self.assertEqual(len(nodes), 2)
        self.assertTrue(all(node['max_len'] == 900 for node in nodes))


if __name__ == '__main__':
    unittest.main()
