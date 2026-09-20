# Публикация AURA-KSP v1.0.0

Готовый сценарий и краткая инструкция находятся в [tools/publishing/START_HERE_RU.md](../tools/publishing/START_HERE_RU.md). В полном ZIP папка `publishing` расположена рядом с `repository` и `research-assets`; в GitHub-клоне та же утилита находится в `tools/publishing`.

Целевой репозиторий — `KuznetsovKarazin/aura-ksp`, тег — `v1.0.0`. Пакет публикуется под **CC BY-NC 4.0** с сохранением сторонних уведомлений из [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Исходный NTU RGB+D не распространяется.

Сценарий для Windows PowerShell 5.1/PowerShell 7 требует Git, GitHub CLI и Python 3.11. Он проверяет manifests, формирует два архива с SHA-256, проверяет Git index и аккаунт, выполняет обычный push, загружает файлы в черновик релиза и проверяет их обратным скачиванием перед публикацией. Параметр `-ReleaseRoot` позволяет указать другой путь к полному пакету. `-PrepareOnly` ограничивает работу локальной проверкой и упаковкой; `-DraftOnly` сохраняет релиз черновиком, но код репозитория уже будет публичным.

Для Zenodo создаётся одна запись с архивами кода и research-assets и двумя SHA-256 sidecar. Реальный DOI добавляется после регистрации. Автоматическую интеграцию GitHub–Zenodo параллельно этому порядку не включают, чтобы не создать вторую запись для того же релиза.

Финансирование, предоставленное авторами:

> This research has been funded by the Science Committee of the Ministry of Science and Higher Education of the Republic of Kazakhstan (Grant No. AP23486538 Research and development of a system for recognizing images in video streams based on artificial intelligence).

Эта формулировка уже включена в release notes, описание Zenodo и `zenodo_metadata.json`. DOI не выдуман: запись ещё должна быть создана и опубликована владельцем аккаунта. В инструкциях и скриптах нет токенов или паролей; GitHub CLI использует вход владельца через браузер.
