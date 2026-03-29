from dataclasses import dataclass, field
from typing import TypedDict, List, Dict, Any, Annotated

@dataclass
class CFG:
    """Единая конфигурация системы"""

    # Настройки подключения к vLLM:
    vllm_base_url: str = "https://vllm.ru.tuna.am/v1"
    vllm_api_key: str = ""  # при необходимости

    # Модели для нодов:
    model_generator_1: str = "Qwen/Qwen3.5-9B"
    model_generator_2: str = "Qwen/Qwen3.5-9B"
    model_aggregator: str = "Qwen/Qwen3.5-9B"  # если будет LLM-агрегатор
    model_validator: str = "Qwen/Qwen3.5-9B"   # если будет LLM-валидатор

    # Температура для каждой ноды (разная для разных задач):
    temperature_generator_1: float = 0.5      # Креативность при извлечении
    temperature_generator_2: float = 0.5      # Креативность при извлечении
    temperature_aggregator: float = 0.6       # Консистентность при объединении
    temperature_validator: float = 0.1        # Строгость при проверке

    # Параметры генерации:
    max_tokens: int = 10_000
    top_k: int = 20

    # Обязательные поля согласно ТЗ:
    required_fields: List[str] = field(default_factory=lambda: [
        "пациент",
        "жалобы",
        "анамнез_заболевания",
        "анамнез_жизни",
        "лабораторные_данные",
        "инструментальные_данные",
        "цель_исследования",
        "источники"
    ])

    # Пороги валидации:
    validation_threshold: float = 0.7
    min_field_length: int = 10

    # Пути к файлам:
    source_file_path: str = "/content/text1_source.txt"

    # Системные промпты:
    system_prompt_1: str = """
Ты — медицинский аналитик-эксперт, специализирующийся на структурировании данных из электронных медицинских карт (ЭМК).

### ЗАДАЧА:
Создай подробное структурированное медицинское резюме на основе предоставленной ЭМК пациента.
Резюме должно быть полезным для любого врача-специалиста, который будет работать с пациентом.

### ПРАВИЛА ИЗВЛЕЧЕНИЯ ДАННЫХ:
1. Формат вывода: Возвращай ТОЛЬКО чистый JSON (без ```json и любых маркеров).
2. Даты: Все даты приводи к формату YYYY-MM-DD.
3. Временное окно:
   - Точка отсчёта = дата последнего документа в ЭМК.
   - Активные данные: за последние 12 месяцев от точки отсчёта.
   - Хронические заболевания, аллергии, перенесённые операции — ВСЕГДА включай, независимо от давности.
4. Конфликты данных: Если информация противоречива — выбирай запись с более поздней датой.
5. Отсутствие данных: Если поле не заполнено в источнике — пиши "Не указано" (не выдумывай!).
6. Цитирование: Для диагнозов и ключевых находок указывай источник: "диагноз (дата: YYYY-MM-DD)".
7. Терминология: Используй стандартную медицинскую терминологию (МКБ-10/11 где применимо).
8. Лаконичность: Избегай избыточных описаний, сохраняй клиническую значимость.

ВАЖНО: для каждого суждения добавляй ссылку [n] по следующему примеру: "суждение [n].".

### JSON-СТРУКТУРА (строго соблюдай):
{
  "пациент": {
    "демография": "Возраст, пол",
    "антропометрия": "Рост, вес, ИМТ (последняя запись)"
  },
  "текущее_обращение": {
    "дата": "YYYY-MM-DD",
    "причина": "Основная жалоба/причина обращения [n]",
    "жалобы": ["Жалоба 1 [n]", "Жалоба 2 [n]"]
  },
  "анамнез_заболевания": {
    "дата_начала": "YYYY-MM-DD или Не указано",
    "динамика": "Краткое описание развития [n]",
    "предварительные_диагнозы": ["Диагноз (дата: YYYY-MM-DD) [n]"],
    "проводимое_лечение": ["Препарат (дозировка, частота) [n]"]
  },
  "анамнез_жизни": {
    "хронические_заболевания": ["Болезнь (дата подтверждения: YYYY-MM-DD) [n]"],
    "госпитализации": ["Причина (период: YYYY-MM-DD — YYYY-MM-DD) [n]"],
    "операции_травмы": ["Описание (дата: YYYY-MM-DD) [n]"],
    "вредные_привычки": ["Описание (стаж, если указано) [n]"],
    "аллергии": ["Аллерген (реакция, если указана) [n]"],
    "семейный_анамнез": "Не указано или описание [n]"
  },
  "лабораторные_данные": [
    {"показатель": "Название", "значение": "Результат [n]", "референс": "Норма [n]", "дата": "YYYY-MM-DD"}
  ],
  "инструментальные_данные": [
    {"исследование": "Название", "дата": "YYYY-MM-DD", "ключевые_находки": "Описание [n]"}
  ],
  "консультации_специалистов": [
    {"специальность": "Врач", "дата": "YYYY-MM-DD", "рекомендации": "Кратко [n]"}
  ],
  "клиническое_резюме": {
    "активные_проблемы": ["Проблема 1 [n]", "Проблема 2 [n]"],
    "требует_внимания": ["Пункт 1 [n]", "Пункт 2 [n]"],
    "рекомендации_для_врача": "Краткий вывод о ключевых рисках и приоритетах"
  },
  "источники": {
    1: "цитата №1",
    2: "цитата №2",
    ...,
    n: "цитата №3"
  }
}

### КРИТИЧЕСКИ ВАЖНО:
- Каждая цитата должна быть не меньше 2-х предложений.
- Каждая цитата должна содержать контекст.
- В каждой цитате ОБЯЗАН содержаться объект цитирования.
- Каждая цитата должна дословно совпадать с фрагментом исходного текста.
"""

    system_prompt_2: str = """
Ты — помощник врача-рентгенолога для подготовки к КТ органов брюшной полости (ОБП).
Твоя задача — составить структурированное медицинское резюме на основе предоставленной информации о пациенте из ЭМК.

### ПРАВИЛА:
Возвращай ТОЛЬКО чистый JSON (без ``` и любых маркеров).
Все даты — в формате YYYY-MM-DD.
Точка отсчёта: дата последнего документа в карте. Фильтруй данные за 1 год от неё.
Если дат нет — включай все данные.
Хронические заболевания и аллергии — ВСЕГДА включай, независимо от давности.
Если данных нет — пиши "Не указано". Если конфликт — бери последнее по дате.
Не выдумывай. Цитируй источник: "диагноз (дата: YYYY-MM-DD)".
В секции "на_что_обратить_внимание" — только патологии ОБП, влияющие на интерпретацию КТ.

ВАЖНО: для каждого суждения добавляй ссылку [n] по следующему примеру: "суждение [n]."

### JSON-СТРУКТУРА:
{
  "пациент": "ФИО, возраст, рост, вес(последняя запись)",
  "жалобы": "Только жалобы, ставшие причиной выполнения КТ [n]",
  "анамнез_заболевания": "Дата начала: YYYY-MM-DD, динамика, диагнозы, лечение [n]",
  "анамнез_жизни": {
    "хронические_заболевания": ["Болезнь (начало: YYYY-MM-DD) [n]"],
    "история болезни": ["Болезнь (год: YYYY-MM-DD) [n]"],
    "вредные_привычки": ["Описание [n]"],
    "операции_травмы": ["Описание (дата: YYYY-MM-DD) [n]"],
    "аллергии": ["Аллерген [n]"]
  },
  "лабораторные_данные": [
    {"показатель": "...[n]", "значение": "...[n]", "дата": "YYYY-MM-DD"}
  ],
  "инструментальные_данные": [
    {"исследование": "...[n]", "дата": "YYYY-MM-DD", "находки": "...[n]"}
  ],
  "цель_исследования": "Краткий вывод, что нужно подтвердить или исключить",
  "на_что_обратить_внимание_для_задачи_КТ_ОБП": ["пункт 1[n]", "пункт 2[n]"]
},
  "источники": {
    1: "цитата №1",
    2: "цитата №2",
    ...,
    n: "цитата №3"
  }

  ### КРИТИЧЕСКИ ВАЖНО:
- Каждая цитата должна быть не меньше 2-х предложений.
- Каждая цитата должна содержать контекст.
- В каждой цитате ОБЯЗАН содержаться объект цитирования.
- Каждая цитата должна дословно совпадать с фрагментом исходного текста.
    """
# Создание глобального экземпляра конфигурации


cfg = CFG()

# # Создание GRAPH


# ## **Graph State** (состояние графа)


!pip install langgraph -q

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from datetime import datetime
import json

from operator import add

class GraphState(TypedDict):
    """
    Состояние графа

    Annotated используется ТОЛЬКО для errors с встроенным operator.add
    timestamps разбиты на отдельные ключи - нет конфликта при параллельной записи
    """
    # Входные данные (только чтение):
    source_emr: str   # Исходная ЭМК

    # Результаты генераторов (каждый пишет в своё поле):
    generator_1_output: Dict[str, Any]
    generator_2_output: Dict[str, Any]

    # Результат агрегации (пишет одна нода):
    aggregated_result: Dict[str, Any]

    # Результат валидации (пишет одна нода):
    validation_result: Dict[str, Any]
    is_valid: bool

    # Метаданные:
    # аnnotated только для errors - параллельные ноды могут добавлять ошибки
    errors: Annotated[List[str], add]
    # отдельные ключи для timestamps - нет конфликта при параллельной записи
    timestamp_gen1: str
    timestamp_gen2: str
    timestamp_aggregator: str
    timestamp_validator: str

# Используйте Annotated только для полей, которые реально требуют объединения (списки ошибок). Остальные поля пусть просто перезаписываются.
# 
# ---
# 
# | Ситуация | Нужно ли `Annotated` | Решение |
# |----------|---------------------|---------|
# | Поле пишет **одна нода** | ❌ Нет | Обычное объявление |
# | Поле пишут **параллельные ноды** (списки) | ✅ Да | `Annotated[List, add]` |
# | Поле пишут **параллельные ноды** (dict) | ✅ Да | Кастомный редюзер **или** разные ключи |
# | Поле только **читается** | ❌ Нет | Обычное объявление |
# 
# ---
# 
# Это стандартный паттерн LangGraph — редюсеры нужны только при реальном конфликте параллельных записей в одно поле.


# ## Helpers (хелперы)
# 
# Вспомогательные (утилитарные) функции


import openai

def create_client(model_name: str) -> openai.OpenAI:
    """Создание клиента для работы с моделью"""
    return openai.OpenAI(
        base_url=cfg.vllm_base_url,
        api_key=cfg.vllm_api_key
    )

def parse_json_from_response(content: str) -> Dict[str, Any]:
    """Парсинг JSON из ответа модели"""
    try:
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end]
            return json.loads(json_str)
        else:
            return {"raw_output": content}
    except json.JSONDecodeError:
        return {"raw_output": content, "parse_error": True}

# def calculate_metrics(result: Dict) -> Dict:
#     """Расчёт итоговых метрик качества"""
#     validation = result.get('validation_result', {})

#     metrics = {
#         'completeness_score': validation.get('checks', {}).get('required_fields_coverage', 0),
#         'accuracy_score': validation.get('checks', {}).get('factual_accuracy', 0),
#         'overall_score': validation.get('score', 0),
#         'is_valid': result.get('is_valid', False),
#         'processing_time': calculate_processing_time(result),
#         'consensus_agreement': calculate_agreement(
#             result.get('generator_1_output', {}),
#             result.get('generator_2_output', {})
#         )
#     }

#     return metrics

# ## **Graph Nodes** (Ноды)
# 
# 4 нода:
# 
# - generator_1
# - generator_2
# - aggregator
# - validator
# 
# Возвращается у каждого Dict с обновляемыми полями (не полный state)


# ```
#            START
#           /     \
#          ↓       ↓
#   generator_1  generator_2
#          \       /
#           ↓     ↓
#         aggregator
#             ↓
#         validator
#             ↓
#            END
# ```
# 
# 
# Возвращать полный state из нод — это неправильный паттерн в LangGraph
# 
# Правильный паттерн LangGraph — возврат только изменённых полей
# 
# Почему возвращать полный `state` — плохая практика:
# 
# ---
# 
# 📊 Сравнение подходов
# 
# | Аспект | Возврат полного `state` | Возврат частичных полей |
# |--------|------------------------|------------------------|
# | **Параллельное выполнение** | ❌ Конфликты | ✅ Работает корректно |
# | **Редюсеры (`Annotated`)** | ❌ Могут ломаться | ✅ Работают как ожидается |
# | **Читаемость кода** | ❌ Непонятно, что меняется | ✅ Явно указаны изменяемые поля |
# | **Риск ошибок** | ❌ Высокий (случайная перезапись) | ✅ Низкий |
# | **Рекомендация LangGraph** | ❌ Не рекомендуется | ✅ Официальный паттерн |
# | **Масштабируемость** | ❌ Плохо | ✅ Хорошо |
# 
# ---
# 
# Официальная документация LangGraph
# 
# Согласно [документации LangGraph](https://langchain-ai.github.io/langgraph/):
# 
# > **"Nodes should return updates to the state, not the entire state."**
# >
# > (Ноды должны возвращать обновления состояния, а не всё состояние целиком.)
# 
# Это **стандартный паттерн** для всех примеров в документации.
# 
# **Никогда не возвращайте полный `state`** — только те поля, которые нода действительно модифицирует. Это официальный паттерн LangGraph, который гарантирует корректную работу с параллельными нодами и редюсерами.


# ### Генератор 1 (Node)


def generate_summary_1(state: GraphState) -> Dict:
    """Генератор 1: Создание суммаризации с первым промптом"""

    try:
        client = create_client(cfg.model_generator_1)
        messages = [
            {"role": "system", "content": cfg.system_prompt_1},
            {"role": "user", "content": f"Суммаризируй медицинский текст:\n\n{state['source_emr']}"}
        ]

        response = client.chat.completions.create(
            model=cfg.model_generator_1,
            messages=messages,
            temperature=cfg.temperature_generator_1,
            max_tokens=cfg.max_tokens,
            extra_body={
                "top_k": cfg.top_k,
                "chat_template_kwargs": {"enable_thinking": False}
            }
        )

        content = response.choices[0].message.content

        parsed_result = parse_json_from_response(content)

        # ВОЗВРАЩАЕТСЯ DICT со своими ключами generator_output и timestamp
        return {
            'generator_1_output': parsed_result,
            'timestamp_gen1': datetime.now().isoformat()
        }

    except Exception as e:
        return {
            'generator_1_output': {"error": str(e)},
            'errors': [f"Generator 1 error: {str(e)}"]
        }


# ### Генератор 2 (Node)


def generate_summary_2(state: GraphState) -> Dict:
    """Генератор 2: Создание суммаризации с альтернативным промптом"""

    try:
        client = create_client(cfg.model_generator_2)
        messages = [
            {"role": "system", "content": cfg.system_prompt_2},
            {"role": "user", "content": f"Проанализируй медицинскую карту:\n\n{state['source_emr']}"}
        ]

        response = client.chat.completions.create(
            model=cfg.model_generator_2,
            messages=messages,
            temperature=cfg.temperature_generator_2,
            max_tokens=cfg.max_tokens,
            extra_body={
                "top_k": cfg.top_k,
                "chat_template_kwargs": {"enable_thinking": False}
            }
        )

        content = response.choices[0].message.content
        parsed_result = parse_json_from_response(content)

        # ВОЗВРАЩАЕТСЯ DICT со своими ключами generator_output и timestamp
        return {
            'generator_2_output': parsed_result,
            'timestamp_gen2': datetime.now().isoformat()
        }

    except Exception as e:
        return {
            'generator_2_output': {"error": str(e)},
            'errors': [f"Generator 2 error: {str(e)}"]
        }

# ### Агрегатор (Node)


def aggregate_results(state: GraphState) -> Dict:
    """Агрегатор: Объединение результатов генераторов по принципу голосования
    """

    aggregated = {}
    gen1 = state.get('generator_1_output', {})
    gen2 = state.get('generator_2_output', {})

    for field_name in cfg.required_fields:
        val1 = gen1.get(field_name, "").strip() if isinstance(gen1.get(field_name), str) else ""
        val2 = gen2.get(field_name, "").strip() if isinstance(gen2.get(field_name), str) else ""

        # Логика голосования
        if val1 and val2:
            # Оба заполнены: выбираем более подробный
            if len(val1) >= len(val2):
                aggregated[field_name] = val1
                aggregated[f"{field_name}_source"] = "generator_1"
            else:
                aggregated[field_name] = val2
                aggregated[f"{field_name}_source"] = "generator_2"
        elif val1:
            aggregated[field_name] = val1
            aggregated[f"{field_name}_source"] = "generator_1"
        elif val2:
            aggregated[field_name] = val2
            aggregated[f"{field_name}_source"] = "generator_2"
        else:
            aggregated[field_name] = ""
            aggregated[f"{field_name}_source"] = "none"

    # Добавляем метаданные
    aggregated['metadata'] = {
        'aggregation_method': 'voting_by_completeness',
        'timestamp': datetime.now().isoformat(),
        'generators_used': ['generator_1', 'generator_2']
    }

    # Возвращатся DICT
    return {
        'aggregated_result': aggregated,
        'timestamp_aggregator': datetime.now().isoformat()
    }

# def aggregate_results(state: GraphState) -> Dict:
#     """LLM-агрегатор для разрешения конфликтов и нормализации JSON"""
#     client = create_client("Qwen/Qwen3.5-9B")

#     messages = [
#         {"role": "system", "content": """Ты — медицинский эксперт-агрегатор.
#         Объедини два варианта суммаризации в один согласованный JSON.
#         Если есть противоречия — выбирай более полный и точный вариант.
#         Верни строго валидный JSON."""},
#         {"role": "user", "content": f"""
#         Вариант 1: {json.dumps(state['generator_1_output'], ensure_ascii=False)}

#         Вариант 2: {json.dumps(state['generator_2_output'], ensure_ascii=False)}

#         Объедини эти данные в один согласованный ответ."""}
#     ]

#     response = client.chat.completions.create(
#         model=cfg.model_aggregator,
#         messages=messages,
#         temperature=cfg.temperature_aggregator,
#         max_tokens=cfg.max_tokens
#     )

#     return {
#         'aggregated_result': parse_json_from_response(response.choices[0].message.content),
#         'timestamp_aggregator': datetime.now().isoformat()
#     }

# ### Валидатор (Node)


def validate_result(state: GraphState) -> Dict:
    """Валидатор: Сверка с исходной ЭМК"""
    print("[DEBUG Validator] Started")
    validation = {
        'checks': {},
        'score': 0.0,
        'is_valid': False,
        'issues': []
    }

    aggregated = state.get('aggregated_result', {})
    source_text = state.get('source_emr', '').lower()

    # 1. Проверка наличия обязательных полей
    required_present = 0
    for field_name in cfg.required_fields:
        value = aggregated.get(field_name, "")
        is_present = bool(value and len(value) > cfg.min_field_length)
        validation['checks'][f'field_{field_name}'] = is_present
        if is_present:
            required_present += 1
        else:
            validation['issues'].append(f"Поле '{field_name}' отсутствует или пустое")

    validation['checks']['required_fields_coverage'] = required_present / len(cfg.required_fields)

    # 2. Проверка фактологической точности
    factual_accuracy = 0.0
    if aggregated.get('жалобы'):
        complaints = aggregated['жалобы'].lower()
        keywords_found = sum(1 for word in complaints.split() if len(word) > 4 and word in source_text)
        total_words = len([w for w in complaints.split() if len(w) > 4])
        factual_accuracy = keywords_found / total_words if total_words > 0 else 0

    validation['checks']['factual_accuracy'] = factual_accuracy
    if factual_accuracy < 0.5:
        validation['issues'].append("Низкая фактологическая точность")

    # 3. Проверка на отсутствие критических пропусков
    critical_info_missing = False
    if 'кт' in source_text or 'компьютерная томография' in source_text:
        if not aggregated.get('инструментальные_данные'):
            critical_info_missing = True
            validation['issues'].append("Отсутствуют данные инструментальных исследований")

    validation['checks']['critical_info_present'] = not critical_info_missing

    # 4. Итоговая оценка
    validation['score'] = (
        validation['checks']['required_fields_coverage'] * 0.4 +
        factual_accuracy * 0.4 +
        (0.2 if not critical_info_missing else 0)
    )

    validation['is_valid'] = validation['score'] >= cfg.validation_threshold and not critical_info_missing
    validation['timestamp'] = datetime.now().isoformat()

    # Возвращатся DICT
    return {
        'validation_result': validation,
        'is_valid': validation['is_valid'],
        'timestamp_validator': datetime.now().isoformat()
    }

# def validate_result(state: GraphState) -> Dict:
#     """LLM-валидатор для проверки фактологической точности"""
#     client = create_client("Qwen/Qwen3.5-9B")

#     messages = [
#         {"role": "system", "content": """Ты — эксперт по контролю качества медицинских суммаризаций.
#         Проверь, соответствует ли суммаризация исходной ЭМК.
#         Оцени по шкале 0-1 и укажи ошибки."""},
#         {"role": "user", "content": f"""
#         Исходная ЭМК: {state['source_emr'][:5000]}...

#         Суммаризация: {json.dumps(state['aggregated_result'], ensure_ascii=False)}

#         Проверь:
#         1. Все ли факты точны?
#         2. Нет ли выдуманной информации?
#         3. Все ли обязательные поля заполнены?

#         Верни JSON: {{"score": 0.0-1.0, "errors": [], "is_valid": true/false}}"""}
#     ]

#     response = client.chat.completions.create(
#         model=cfg.model_validator,
#         messages=messages,
#         temperature=cfg.temperature_validator,
#         max_tokens=cfg.max_tokens
#     )

#     validation = parse_json_from_response(response.choices[0].message.content)

#     return {
#         'validation_result': validation,
#         'is_valid': validation.get('is_valid', False),
#         'timestamp_validator': datetime.now().isoformat()
#     }

# ## Инициализация графа


def create_voting_graph() -> StateGraph:
    """Создание графа с архитектурой голосования (Параллельные/независимые генраторы)"""

    workflow = StateGraph(GraphState)

    # Добавление нод (Nodes):
    workflow.add_node("generator_1", generate_summary_1)
    workflow.add_node("generator_2", generate_summary_2)
    workflow.add_node("aggregator", aggregate_results)
    workflow.add_node("validator", validate_result)

    # Добавляем рёбра (Edges):
    workflow.add_edge(START, "generator_1")
    workflow.add_edge(START, "generator_2")
    workflow.add_edge("generator_1", "aggregator")
    workflow.add_edge("generator_2", "aggregator")
    workflow.add_edge("aggregator", "validator")
    workflow.add_edge("validator", END)

    return workflow.compile()


# 
# 
# ---
# 
# 
# 
# Визуализация графа


from IPython.display import Image, display

# Создаем граф
graph = create_voting_graph()

# Визуализируем
display(Image(graph.get_graph().draw_mermaid_png()))

# ## Функция для запуска


def run_summarization_pipeline(source_emr_text: str) -> Dict[str, Any]:
    """
    Запуск пайплайна суммаризации

    Args:
        source_emr_text: Исходный текст ЭМК

    Returns:
        Словарь с результатами
    """
    # Инициализация графа
    app = create_voting_graph()

    # Начальное состояние
    initial_state = {
        "source_emr": source_emr_text,
        "generator_1_output": {},
        "generator_2_output": {},
        "aggregated_result": {},
        "validation_result": {},
        "is_valid": False,
        "errors": [],
        "timestamp_gen1": "",
        "timestamp_gen2": "",
        "timestamp_aggregator": "",
        "timestamp_validator": ""
    }

    # Запуск
    result = app.invoke(initial_state)

    return result

# # Запуск GRAPH (Pipeline)


if __name__ == "__main__":
    summorized_text = """"""

    # Чтение исходного текста из файла
    with open(cfg.source_file_path, "r", encoding="utf-8") as f:
        source_text = f.read()

    summorized_text += "-" * 80

    # Запуск пайплайна
    result = run_summarization_pipeline(source_text)

    # Вывод результатов
    summorized_text += "\n📋 РЕЗУЛЬТАТЫ ГЕНЕРАТОРА 1:"
    summorized_text += json.dumps(result.get('generator_1_output', {}), ensure_ascii=False, indent=2)

    summorized_text += "\n📋 РЕЗУЛЬТАТЫ ГЕНЕРАТОРА 2:"
    summorized_text += json.dumps(result.get('generator_2_output', {}), ensure_ascii=False, indent=2)

    # summorized_text += "\n🔄 АГРЕГИРОВАННЫЙ РЕЗУЛЬТАТ:"
    # summorized_text += json.dumps(result.get('aggregated_result', {}), ensure_ascii=False, indent=2)

    # summorized_text += "\n✅ РЕЗУЛЬТАТ ВАЛИДАЦИИ:"
    # summorized_text += json.dumps(result.get('validation_result', {}), ensure_ascii=False, indent=2)

    # summorized_text += f"\n🎯 ИТОГОВАЯ ВАЛИДНОСТЬ: {'✓ ПРОЙДЕНО' if result.get('is_valid') else '✗ НЕ ПРОЙДЕНО'}"
    # summorized_text += f"📊 Оценка качества: {result.get('validation_result', {}).get('score', 0):.2f}"

    # print("\n⏱️ ВРЕМЕННЫЕ МЕТКИ:")
    # print(f"  Генератор 1: {result.get('timestamp_gen1', 'N/A')}")
    # print(f"  Генератор 2: {result.get('timestamp_gen2', 'N/A')}")
    # print(f"  Агрегатор:   {result.get('timestamp_aggregator', 'N/A')}")
    # print(f"  Валидатор:   {result.get('timestamp_validator', 'N/A')}")

    if result.get('errors'):
        print("\n⚠️ ОШИБКИ:")
        for error in result['errors']:
            print(f"  - {error}")

    with open("summorized.txt", "w", encoding="utf-8") as file:
      file.write(summorized_text)