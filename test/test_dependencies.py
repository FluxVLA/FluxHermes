"""Keep installed and requirements-file dependencies consistent."""
import configparser
import unittest
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


class DependenciesTest(unittest.TestCase):

    def test_onnx_is_a_base_dependency_for_bundle_inspection(self):
        config = configparser.ConfigParser()
        config.read(ROOT / 'setup.cfg')
        installed = {
            item.name: str(item.specifier)
            for line in config['options']['install_requires'].splitlines() if line
            for item in [Requirement(line)]
        }
        requirements = {
            item.name: str(item.specifier)
            for line in (ROOT / 'requirements-base.txt').read_text().splitlines()
            if line and not line.startswith('#')
            for item in [Requirement(line)]
        }
        self.assertIn('onnx', installed)
        self.assertEqual(requirements, installed)
        for extra in ('runtime', 'export'):
            lines = (ROOT / f'requirements-{extra}.txt').read_text().splitlines()
            self.assertIn('-r requirements-base.txt', lines)


if __name__ == '__main__':
    unittest.main()
