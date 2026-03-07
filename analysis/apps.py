from django.apps import AppConfig


class AnalysisConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'analysis'

    def ready(self):
        import analysis.signals
        from .utils import ensure_core_groups
        try:
            ensure_core_groups()
        except Exception:
            pass
