# VaquitasDeColores_Hackathon_2026_UniCaldas

Copiloto analitico para consultas sobre comportamiento web, construido con:

- `Motor_Analitico/` para limpieza y generacion de datasets analiticos
- `analytics_core/` para carga, mapping y metricas reutilizables
- `LLM_Workflow/` para intents, prompts y respuestas con Gemini
- `App_Streamlit/` para la interfaz conversacional
  Los componentes de UI viven aqui, pero el entrypoint oficial es `app.py` en la raiz.

## Preparacion

1. Instala dependencias:

```bash 
pip install -r requirements.txt
```

2. Configura variables de entorno en un archivo `.env`:

```env
GEMINI_API_KEY=tu_api_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.2
```

3. Genera o verifica los outputs analiticos:

```bash
python Motor_Analitico/Data_cleaning.py
python Motor_Analitico/Flujo_navegacion.py
```

## Ejecutar la app

```bash
streamlit run app.py
```

## Ejecutar por CLI

```bash
python LLM_Workflow/LLM_call.py "Cual fue la pagina mas vista hoy?"
```

## Que soporta la v1

- Pagina o producto mas visto
- Porcentaje de sesiones por pagina o producto
- Pagina de salida principal
- Flujo de navegacion mas frecuente
- Rebote o abandono rapido
- Comparacion por pais, dispositivo o navegador
- Tendencia temporal basica

## Notas

- Si no hay `GEMINI_API_KEY`, el sistema sigue funcionando con una respuesta deterministica de respaldo.
- La integracion del proveedor LLM esta preparada para Gemini mediante `google-genai`.
