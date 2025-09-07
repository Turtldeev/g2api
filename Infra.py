# DeepInfra (79% original models)
from __future__ import annotations
import requests
from typing import List, Dict, Any, Union

class DeepInfra:
    # URLs
    url = "https://deepinfra.com"
    login_url = "https://deepinfra.com/dash/api_keys"
    api_base = "https://api.deepinfra.com/v1/openai"
    models_url = "https://api.deepinfra.com/models/featured"

    # Статусы
    working = True
    active_by_default = True

    # Списки моделей (будут заполнены динамически)
    models: List[str] = []
    image_models: List[str] = []
    vision_models: List[str] = []
    live = 0  # счетчик успешных обновлений списка моделей

    # Алиасы моделей
    model_aliases: Dict[str, Union[str, List[str]]] = {
        # cognitivecomputations
        "dolphin-2.6": "cognitivecomputations/dolphin-2.6-mixtral-8x7b",
        "dolphin-2.9": "cognitivecomputations/dolphin-2.9.1-llama-3-70b",
        # deepinfra
        "airoboros-70b": "deepinfra/airoboros-70b",
        # deepseek-ai
        "deepseek-prover-v2": "deepseek-ai/DeepSeek-Prover-V2-671B",
        "deepseek-prover-v2-671b": "deepseek-ai/DeepSeek-Prover-V2-671B",
        "deepseek-r1": ["deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-R1-0528"],
        "deepseek-r1-0528": "deepseek-ai/DeepSeek-R1-0528",
        "deepseek-r1-0528-turbo": "deepseek-ai/DeepSeek-R1-0528-Turbo",
        "deepseek-r1-distill-llama-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-r1-distill-qwen-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-r1-turbo": "deepseek-ai/DeepSeek-R1-Turbo",
        "deepseek-v3": ["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3-0324"],
        "deepseek-v3-0324": "deepseek-ai/DeepSeek-V3-0324",
        "deepseek-v3-0324-turbo": "deepseek-ai/DeepSeek-V3-0324-Turbo",
        # google
        "codegemma-7b": "google/codegemma-7b-it",
        "gemma-1.1-7b": "google/gemma-1.1-7b-it",
        "gemma-2-27b": "google/gemma-2-27b-it",
        "gemma-2-9b": "google/gemma-2-9b-it",
        "gemma-3-4b": "google/gemma-3-4b-it",
        "gemma-3-12b": "google/gemma-3-12b-it",
        "gemma-3-27b": "google/gemma-3-27b-it",
        # lizpreciatior
        "lzlv-70b": "lizpreciatior/lzlv_70b_fp16_hf",
        # meta-llama
        "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama-3.2-90b": "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
        "llama-4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "llama-4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        # microsoft
        "phi-4": "microsoft/phi-4",
        "phi-4-multimodal": "microsoft/Phi-4-multimodal-instruct",
        "phi-4-reasoning-plus": "microsoft/phi-4-reasoning-plus",
        "wizardlm-2-7b": "microsoft/WizardLM-2-7B",
        "wizardlm-2-8x22b": "microsoft/WizardLM-2-8x22B",
        # mistralai
        "mistral-small-3.1-24b": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        # Qwen
        "qwen-3-14b": "Qwen/Qwen3-14B",
        "qwen-3-30b": "Qwen/Qwen3-30B-A3B",
        "qwen-3-32b": "Qwen/Qwen3-32B",
        "qwen-3-235b": "Qwen/Qwen3-235B-A22B",
        "qwq-32b": "Qwen/QwQ-32B",
    }

    # Визуальные модели (можно расширять)
    default_vision_model = "meta-llama/Llama-3.2-90B-Vision-Instruct"
    vision_models = [
        default_vision_model,
        'meta-llama/Llama-3.2-90B-Vision-Instruct',
        'openai/gpt-oss-120b',
        'openai/gpt-oss-20b',
    ]

    @classmethod
    def get_models(cls, **kwargs) -> List[str]:
        """Получает список моделей с DeepInfra API и кэширует его."""
        if not cls.models:
            try:
                response = requests.get(cls.models_url, timeout=10)
                response.raise_for_status()
                models_data = response.json()

                cls.models.clear()
                cls.image_models.clear()

                for model in models_data:
                    model_name = model.get('model_name')
                    if not model_name:
                        continue
                    model_type = model.get("type")
                    reported_type = model.get("reported_type")

                    if model_type == "text-generation":
                        cls.models.append(model_name)
                    elif reported_type == "text-to-image":
                        cls.image_models.append(model_name)

                # Объединяем текстовые и изобразительные модели
                cls.models.extend(cls.image_models)

                if models_data:
                    cls.live += 1

            except Exception as e:
                print(f"Ошибка при получении моделей: {e}")
                # Не прерываем работу — возвращаем пустой или предыдущий список
                return cls.models

        return cls.models

    @classmethod
    def resolve_model(cls, alias: str) -> str:
        """Разрешает алиас модели в полное имя."""
        mapping = cls.model_aliases.get(alias)
        if isinstance(mapping, list):
            # Берем первый вариант по умолчанию
            return mapping[0]
        elif isinstance(mapping, str):
            return mapping
        else:
            # Если алиас не найден, возвращаем как есть
            return alias

    @classmethod
    def is_vision_model(cls, model_name: str) -> bool:
        """Проверяет, является ли модель визуальной."""
        resolved = cls.resolve_model(model_name)
        return resolved in cls.vision_models
