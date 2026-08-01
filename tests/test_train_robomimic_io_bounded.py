import unittest
from types import SimpleNamespace

from scripts import train_robomimic_io_bounded as module


class IoBoundHooksTests(unittest.TestCase):
    def test_skips_only_resume_artifacts_and_restores_original_functions(self):
        save_calls = []
        copy_calls = []

        def save_model(*args, **kwargs):
            save_calls.append((args, kwargs))

        def copyfile(source, destination):
            copy_calls.append((source, destination))
            return destination

        train_module = SimpleNamespace(
            TrainUtils=SimpleNamespace(save_model=save_model),
            shutil=SimpleNamespace(copyfile=copyfile),
        )

        with module.suppress_resume_artifact_writes(train_module):
            train_module.TrainUtils.save_model(ckpt_path="/run/last.pth")
            train_module.TrainUtils.save_model(ckpt_path="/run/models/model_epoch_10.pth")
            result = train_module.shutil.copyfile(
                "/run/last.pth",
                "/run/last_bak.pth",
            )

        self.assertEqual(len(save_calls), 1)
        self.assertEqual(
            save_calls[0][1]["ckpt_path"],
            "/run/models/model_epoch_10.pth",
        )
        self.assertEqual(copy_calls, [])
        self.assertEqual(result, "/run/last_bak.pth")
        self.assertIs(train_module.TrainUtils.save_model, save_model)
        self.assertIs(train_module.shutil.copyfile, copyfile)


if __name__ == "__main__":
    unittest.main()
