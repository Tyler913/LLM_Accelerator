from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import vitis


RUNTIME_SOURCE_FILES = (
    "main.c",
    "qmap_one_token_runtime.h",
    "qmap_one_token_regs.h",
    "qmap_model_config_generated.h",
)

DEMO_SOURCE_FILES = (
    "qot_session.c",
    "qot_session.h",
    "qot_protocol.c",
    "qot_protocol.h",
    "qot_uart.c",
    "qot_uart.h",
)

TOKENIZER_RUNTIME_SOURCE_FILES = (
    "qtk_tokenizer_runtime.c",
    "qtk_tokenizer_runtime.h",
    "qtk_text_tokenizer.c",
    "qtk_text_tokenizer.h",
)


def require_path_from_environment(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must name an existing file")
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def populate_application(
    component_dir: Path, source_dir: Path, *, model_smoke: bool
) -> None:
    app_source_dir = component_dir / "src"
    if not app_source_dir.is_dir():
        raise FileNotFoundError(app_source_dir)
    for name in RUNTIME_SOURCE_FILES:
        shutil.copy2(source_dir / name, app_source_dir / name)

    if model_smoke:
        user_config = app_source_dir / "UserConfig.cmake"
        text = user_config.read_text(encoding="utf-8")
        old = 'set(USER_COMPILE_DEFINITIONS\n""\n)'
        new = 'set(USER_COMPILE_DEFINITIONS\n"QOT_MODEL_BOARD_SMOKE=1"\n)'
        if old not in text:
            raise RuntimeError(
                f"{user_config}: could not locate USER_COMPILE_DEFINITIONS"
            )
        user_config.write_text(text.replace(old, new, 1), encoding="utf-8")


def populate_generate_application(
    component_dir: Path,
    source_dir: Path,
    demo_source_dir: Path,
    tokenizer_asset: Path,
) -> None:
    app_source_dir = component_dir / "src"
    if not app_source_dir.is_dir():
        raise FileNotFoundError(app_source_dir)

    for name in RUNTIME_SOURCE_FILES[1:]:
        shutil.copy2(source_dir / name, app_source_dir / name)
    shutil.copy2(demo_source_dir / "main_generate.c", app_source_dir / "main.c")
    for name in DEMO_SOURCE_FILES:
        shutil.copy2(demo_source_dir / name, app_source_dir / name)
    tokenizer_runtime_dir = demo_source_dir / "tokenizer_runtime"
    for name in TOKENIZER_RUNTIME_SOURCE_FILES:
        shutil.copy2(tokenizer_runtime_dir / name, app_source_dir / name)

    copied_asset = app_source_dir / "qwen3_tokenizer.qtk"
    shutil.copy2(tokenizer_asset, copied_asset)
    assembly_path = app_source_dir / "tokenizer_asset.S"
    assembly_path.write_text(
        '.section .rodata.qot_tokenizer_asset,"a",%progbits\n'
        ".balign 64\n"
        ".global qot_tokenizer_asset_start\n"
        ".type qot_tokenizer_asset_start, %object\n"
        "qot_tokenizer_asset_start:\n"
        f'.incbin "{copied_asset.as_posix()}"\n'
        ".global qot_tokenizer_asset_end\n"
        "qot_tokenizer_asset_end:\n"
        ".size qot_tokenizer_asset_start, "
        "qot_tokenizer_asset_end - qot_tokenizer_asset_start\n"
        '.section .note.GNU-stack,"",%progbits\n',
        encoding="utf-8",
    )

    user_config = app_source_dir / "UserConfig.cmake"
    text = user_config.read_text(encoding="utf-8")
    new = (
        'set(USER_COMPILE_SOURCES\n'
        '"main.c"\n'
        '"qot_session.c"\n'
        '"qot_protocol.c"\n'
        '"qot_uart.c"\n'
        '"qtk_tokenizer_runtime.c"\n'
        '"qtk_text_tokenizer.c"\n'
        '"tokenizer_asset.S"\n'
        ')'
    )
    source_block = re.compile(
        r"^set\(USER_COMPILE_SOURCES[ \t]*\r?\n.*?^\)",
        flags=re.MULTILINE | re.DOTALL,
    )
    text, replacement_count = source_block.subn(new, text, count=1)
    if replacement_count != 1:
        raise RuntimeError(
            f"{user_config}: could not locate USER_COMPILE_SOURCES"
        )
    user_config.write_text(text, encoding="utf-8")


def configure_hardware_launch(
    component_dir: Path, *, stop_at_entry: bool
) -> None:
    launch_path = component_dir / "_ide" / "launch.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    configurations = launch.get("configurations", [])
    if not configurations:
        raise RuntimeError(f"{launch_path}: no hardware launch configuration")
    for configuration in configurations:
        target_setup = configuration["targetSetup"]
        target_setup["zuInitialization"]["usingPsuInit"]["runPsuInit"] = False
        for download in target_setup.get("downloadElf", []):
            download["stopAtEntry"] = stop_at_entry
    launch_path.write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    xsa_path = require_path_from_environment("QOT_XSA")
    tokenizer_asset = require_path_from_environment("QOT_TOKENIZER_ASSET")
    workspace = Path(os.environ.get("QOT_VITIS_WORKSPACE", r"F:\vws")).resolve()
    platform_name = os.environ.get("QOT_VITIS_PLATFORM", "p_qot")
    control_app_name = os.environ.get("QOT_VITIS_CONTROL_APP", "a_qctl")
    model_app_name = os.environ.get("QOT_VITIS_MODEL_APP", "a_qmdl")
    generate_app_name = os.environ.get("QOT_VITIS_GENERATE_APP", "a_qgen")
    source_dir = Path(__file__).resolve().parent
    demo_source_dir = source_dir.parent / "qmap_prompt_demo"

    workspace.mkdir(parents=True, exist_ok=True)
    client = vitis.create_client()
    try:
        client.set_workspace(path=workspace.as_posix())
        advanced_options = client.create_advanced_options_dict(dt_overlay="0")
        created_platform = client.create_platform_component(
            name=platform_name,
            hw_design=str(xsa_path),
            os="standalone",
            cpu="psu_cortexa53_0",
            domain_name="standalone_psu_cortexa53_0",
            generate_dtb=False,
            advanced_options=advanced_options,
            architecture="64-bit",
            compiler="gcc",
        )
        print(f"platform_create={created_platform}")
        platform = client.get_component(name=platform_name)
        print(f"platform_build={platform.build()}")

        xpfm = (
            f"$COMPONENT_LOCATION/../{platform_name}/export/"
            f"{platform_name}/{platform_name}.xpfm"
        )
        created_control_app = client.create_app_component(
            name=control_app_name,
            platform=xpfm,
            domain="standalone_psu_cortexa53_0",
        )
        print(f"control_app_create={created_control_app}")
        control_app = client.get_component(name=control_app_name)
        populate_application(
            workspace / control_app_name, source_dir, model_smoke=False
        )
        configure_hardware_launch(
            workspace / control_app_name, stop_at_entry=False
        )
        print(f"control_app_build={control_app.build()}")

        created_model_app = client.create_app_component(
            name=model_app_name,
            platform=xpfm,
            domain="standalone_psu_cortexa53_0",
        )
        print(f"model_app_create={created_model_app}")
        model_app = client.get_component(name=model_app_name)
        populate_application(workspace / model_app_name, source_dir, model_smoke=True)
        configure_hardware_launch(workspace / model_app_name, stop_at_entry=True)
        print(f"model_app_build={model_app.build()}")

        created_generate_app = client.create_app_component(
            name=generate_app_name,
            platform=xpfm,
            domain="standalone_psu_cortexa53_0",
        )
        print(f"generate_app_create={created_generate_app}")
        generate_app = client.get_component(name=generate_app_name)
        populate_generate_application(
            workspace / generate_app_name,
            source_dir,
            demo_source_dir,
            tokenizer_asset,
        )
        configure_hardware_launch(
            workspace / generate_app_name, stop_at_entry=True
        )
        print(f"generate_app_build={generate_app.build()}")
    finally:
        vitis.dispose()

    print(f"workspace={workspace}")
    print(f"platform={workspace / platform_name}")
    print(f"control_elf={workspace / control_app_name / 'build' / (control_app_name + '.elf')}")
    print(f"model_elf={workspace / model_app_name / 'build' / (model_app_name + '.elf')}")
    print(f"generate_elf={workspace / generate_app_name / 'build' / (generate_app_name + '.elf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
