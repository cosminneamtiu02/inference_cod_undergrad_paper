import os
import typing
import yaml
import boto3


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton,
                                        cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class AppConfig(metaclass=Singleton):
    """
    Singleton class for app configuration
    Adapted from https://charlesreid1.github.io/a-singleton-configuration-class-in-python.html
    """

    _LOCAL = False
    _CONFIG_FILE: typing.Optional[str] = None
    _CONFIG: typing.Optional[dict] = None
    _S3_CONFIG: typing.Optional[dict] = None
    _LOCAL_CONFIG: typing.Optional[dict] = None

    _S3_SESSION = None
    _S3_RESOURCE = None
    _S3_BUCKET = None
    _S3_CLIENT = None

    def __init__(self, config_file=None, for_local=False):
        if not config_file:
            raise Exception("Config file not set")

        AppConfig._CONFIG_FILE = config_file

        # load config file
        with open(config_file, 'r') as f:
            AppConfig._CONFIG = yaml.load(f, Loader=yaml.FullLoader)

        boto3.setup_default_session(region_name=os.getenv("AWS_REGION", "eu-west-3"))

        # load config for s3 or local
        if for_local:
            AppConfig._LOCAL_CONFIG = AppConfig._CONFIG.get('local', None)
            AppConfig._LOCAL = True
        else:
            AppConfig._S3_CONFIG = AppConfig._CONFIG.get('s3', None)
            AppConfig._S3_RESOURCE = boto3.resource('s3')
            AppConfig._S3_BUCKET = AppConfig._S3_RESOURCE.Bucket(AppConfig.get_config('bucket_name'))

            # profile = AppConfig._S3_CONFIG.get('profile', None)
            region = AppConfig._S3_CONFIG.get('region', None)

            AppConfig._S3_CLIENT = boto3.client('s3', region_name=region)

    @staticmethod
    def is_local() -> bool:
        return AppConfig._LOCAL

    @staticmethod
    def get_config(config_var: str) -> str:
        config = AppConfig._LOCAL_CONFIG if AppConfig._LOCAL \
                 else AppConfig._S3_CONFIG

        result = config.get(config_var, None)

        return result

    @staticmethod
    def get_s3_object(s3_object):
        if AppConfig.is_local():
            return None

        return s3_object

    @staticmethod
    def get_s3_resource():
        return AppConfig.get_s3_object(AppConfig._S3_RESOURCE)

    @staticmethod
    def get_s3_bucket():
        return AppConfig.get_s3_object(AppConfig._S3_BUCKET)

    @staticmethod
    def get_s3_client():
        return AppConfig.get_s3_object(AppConfig._S3_CLIENT)

    @staticmethod
    def get_input_path() -> str:
        return AppConfig.get_config('input_path')

    @staticmethod
    def get_output_path() -> str:
        return AppConfig.get_config('output_path')
