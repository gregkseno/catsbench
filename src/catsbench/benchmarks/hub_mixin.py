import inspect
import json
import os
from dataclasses import Field, asdict, is_dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Protocol, Type, TypeVar, Union, override

from huggingface_hub import constants
from huggingface_hub.hub_mixin import PyTorchModelHubMixin
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils import (
    SoftTemporaryDirectory,
    logging,
    validate_hf_hub_args,
)


logger = logging.get_logger(__name__)
T = TypeVar('T', bound='PyTorchModelHubMixin')

class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, Field]]

class BenchmarkModelHubMixin(PyTorchModelHubMixin):
    '''
    Implementation of `PyTorchModelHubMixin` to provide Hub upload/download capabilities to Categorical Schrödinger Bridge Benchmark.
    '''
    
    @override
    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        subfolder: Optional[Union[str, Path]] = None,
        config: Optional[Union[dict, DataclassInstance]] = None,
        repo_id: Optional[str] = None,
        push_to_hub: bool = False,
        **push_to_hub_kwargs,
    ) -> Optional[str]:
        '''
        Save benchmark in local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the checkpoint and configuration will be saved.
            subfolder (`str` or `Path`, *optional*):
                Sub-directory inside `save_directory` where the checkpoint and configuration will be saved.
            config (`dict` or `DataclassInstance`, *optional*):
                Benchmark configuration specified as a key/value dictionary or a dataclass instance.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your benchmark to the Hugging Face Hub after saving it.
            repo_id (`str`, *optional*):
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to the folder name if
                not provided.
            push_to_hub_kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the `~ModelHubMixin.push_to_hub` method.

        Returns:
            `str` or `None`: url of the commit on the Hub if `push_to_hub=True`, `None` otherwise.
        '''
        save_directory = Path(save_directory)
        if subfolder is not None:
            save_directory = save_directory / Path(subfolder)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Remove config.json if already exists. After `_save_pretrained` we don't want to overwrite config.json
        # as it might have been saved by the custom `_save_pretrained` already. However we do want to overwrite
        # an existing config.json if it was not saved by `_save_pretrained`.
        config_path = save_directory / constants.CONFIG_NAME
        config_path.unlink(missing_ok=True)

        # save model weights/files (framework-specific)
        self._save_pretrained(save_directory)

        # save config (if provided and if not serialized yet in `_save_pretrained`)
        if config is None:
            config = self._hub_mixin_config
        if config is not None:
            if is_dataclass(config):
                config = asdict(config)  # type: ignore[arg-type]
            if not config_path.exists():
                config_str = json.dumps(config, sort_keys=True, indent=2)
                config_path.write_text(config_str)

        # push to the Hub if required
        if push_to_hub:
            kwargs = push_to_hub_kwargs.copy()  # soft-copy to avoid mutating input
            if config is not None:  # kwarg for `push_to_hub`
                kwargs['config'] = config
            if repo_id is None:
                repo_id = save_directory.name  # Defaults to `save_directory` name
            return self.push_to_hub(repo_id=repo_id, subfolder=subfolder, **kwargs)
        return None

    @validate_hf_hub_args
    @override
    def push_to_hub(
        self,
        repo_id: str,
        *,
        subfolder: Optional[Union[str, Path]] = None,
        config: Optional[Union[dict, DataclassInstance]] = None,
        commit_message: str = 'Push benchmark using huggingface_hub.',
        private: Optional[bool] = None,
        token: Optional[str] = None,
        branch: Optional[str] = None,
        create_pr: Optional[bool] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        delete_patterns: Optional[Union[List[str], str]] = None,
    ) -> str:
        '''
        Upload benchmark to the Hub.

        Use `allow_patterns` and `ignore_patterns` to precisely filter which files should be pushed to the Hub. Use
        `delete_patterns` to delete existing remote files in the same commit.

        Args:
            repo_id (`str`):
                ID of the repository to push to (example: `'username/my-benchmark'`).
            subfolder (`str` or `Path`, *optional*):
                Path in the repository where the benchmark files will be stored.
            config (`dict` or `DataclassInstance`, *optional*):
                Benchmark configuration specified as a key/value dictionary or a dataclass instance.
            commit_message (`str`, *optional*):
                Message to commit while pushing.
            private (`bool`, *optional*):
                Whether the repository created should be private.
                If `None` (default), the repo will be public unless the organization's default is private.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `hf auth login`.
            branch (`str`, *optional*):
                The git branch on which to push the model. This defaults to `'main'`.
            create_pr (`bool`, *optional*):
                Whether or not to create a Pull Request from `branch` with that commit. Defaults to `False`.
            allow_patterns (`List[str]` or `str`, *optional*):
                If provided, only files matching at least one pattern are pushed.
            ignore_patterns (`List[str]` or `str`, *optional*):
                If provided, files matching any of the patterns are not pushed.
            delete_patterns (`List[str]` or `str`, *optional*):
                If provided, remote files matching any of the patterns will be deleted from the repo.

        Returns:
            `str`: The url of the commit of your benchmark in the given repository.
        '''
        api = HfApi(token=token)
        repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id

        # Push the files to the repo in a single commit
        with SoftTemporaryDirectory() as tmp:
            saved_path = Path(tmp) / repo_id
            self.save_pretrained(saved_path, config=config)
            return api.upload_folder(
                repo_id=repo_id,
                repo_type='model',
                folder_path=saved_path,
                path_in_repo=str(subfolder) if subfolder else None,
                commit_message=commit_message,
                revision=branch,
                create_pr=create_pr,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                delete_patterns=delete_patterns,
            )

    @classmethod
    @validate_hf_hub_args
    @override
    def from_pretrained(
        cls: Type[T],
        pretrained_model_name_or_path: Union[str, Path],
        subfolder: Union[str, Path],
        *,
        force_download: bool = False,
        resume_download: Optional[bool] = None,
        proxies: Optional[Dict] = None,
        token: Optional[Union[str, bool]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        **model_kwargs,
    ) -> T:
        '''
        Download a benchmark from the Hugging Face Hub and instantiate it.

        Args:
            pretrained_model_name_or_path (`str`, `Path`):
                - Either the `model_id` (string) of a benchmark hosted on the Hub, e.g. `bigscience/bloom`.
                - Or a path to a `directory` containing checkpoints saved using
                    `.save_pretrained`, e.g., `../path/to/my_model_directory/`.
            subfolder (`str`, `Path`):
                A subfolder inside the model repository that contains the checkpoint and configuration files.
            revision (`str`, *optional*):
                Revision of the benchmark on the Hub. Can be a branch name, a git tag or any commit id.
                Defaults to the latest commit on `main` branch.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to force (re-)downloading the checkpoint and configuration files from the Hub, overriding
                the existing cache.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on every request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `hf auth login`.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            local_files_only (`bool`, *optional*, defaults to `False`):
                If `True`, avoid downloading the file and return the path to the local cached file if it exists.
            model_kwargs (`Dict`, *optional*):
                Additional kwargs to pass to the benchmark during initialization.

        Returns:
            `T`: An instance of the benchmark class.
        '''
        model_id = str(pretrained_model_name_or_path)
        subfolder = str(subfolder)
        model_dir = os.path.join(model_id, subfolder)

        config_file: Optional[str] = None
        if os.path.isdir(model_dir):
            if constants.CONFIG_NAME in os.listdir(model_dir):
                config_file = os.path.join(model_dir, constants.CONFIG_NAME)
            else:
                logger.warning(f'{constants.CONFIG_NAME} not found in {Path(model_dir).resolve()}')
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=constants.CONFIG_NAME,
                    subfolder=subfolder,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                logger.info(f'{constants.CONFIG_NAME} not found on the HuggingFace Hub: {str(e)}')

        # Read config
        config = None
        if config_file is not None:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Decode custom types in config
            for key, value in config.items():
                if key in cls._hub_mixin_init_parameters:
                    expected_type = cls._hub_mixin_init_parameters[key].annotation
                    if expected_type is not inspect.Parameter.empty:
                        config[key] = cls._decode_arg(expected_type, value)

            # Populate model_kwargs from config
            for param in cls._hub_mixin_init_parameters.values():
                if param.name not in model_kwargs and param.name in config:
                    model_kwargs[param.name] = config[param.name]

            # Check if `config` argument was passed at init
            if 'config' in cls._hub_mixin_init_parameters and 'config' not in model_kwargs:
                # Decode `config` argument if it was passed
                config_annotation = cls._hub_mixin_init_parameters['config'].annotation
                config = cls._decode_arg(config_annotation, config)

                # Forward config to model initialization
                model_kwargs['config'] = config

            # Inject config if `**kwargs` are expected
            if is_dataclass(cls):
                for key in cls.__dataclass_fields__:
                    if key not in model_kwargs and key in config:
                        model_kwargs[key] = config[key]
            elif any(param.kind == inspect.Parameter.VAR_KEYWORD for param in cls._hub_mixin_init_parameters.values()):
                for key, value in config.items():
                    if key not in model_kwargs:
                        model_kwargs[key] = value

            # Finally, also inject if `_from_pretrained` expects it
            if cls._hub_mixin_inject_config and 'config' not in model_kwargs:
                model_kwargs['config'] = config

        instance = cls._from_pretrained(
            model_id=str(model_id),
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            **model_kwargs,
        )

        # Implicitly set the config as instance attribute if not already set by the class
        # This way `config` will be available when calling `save_pretrained` or `push_to_hub`.
        if config is not None and (getattr(instance, '_hub_mixin_config', None) in (None, {})):
            instance._hub_mixin_config = config

        return instance
    
    @classmethod
    @override
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        subfolder: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = 'cpu',
        strict: bool = False,
        **model_kwargs,
    ):
        '''Load Pytorch pretrained weights and return the loaded model.'''
        model = cls(**model_kwargs)
        model_dir = os.path.join(model_id, subfolder)
        if os.path.isdir(model_dir):
            print('Loading weights from local directory')
            model_file = os.path.join(model_dir, constants.SAFETENSORS_SINGLE_FILE)
            return cls._load_as_safetensor(model, model_file, map_location, strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    subfolder=subfolder,
                    filename=constants.SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                return cls._load_as_safetensor(model, model_file, map_location, strict)
            except EntryNotFoundError:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    subfolder=subfolder,
                    filename=constants.PYTORCH_WEIGHTS_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                return cls._load_as_pickle(model, model_file, map_location, strict)
