# Опубликовать AURA-KSP на GitHub и Zenodo

Используйте окончательный пакет `AURA-KSP_PUBLIC_RELEASE_v1.0.0`. Старые `rc1` и архивы переходов загружать не нужно. Авторы уже заполнены; Oleksandr Kuznetsov указан вторым.

## GitHub: одна команда после входа

Распакуйте пакет в `E:\AURA`, чтобы появились:

- `E:\AURA\AURA-KSP_PUBLIC_RELEASE_v1.0.0\repository`
- `E:\AURA\AURA-KSP_PUBLIC_RELEASE_v1.0.0\research-assets`
- `E:\AURA\AURA-KSP_PUBLIC_RELEASE_v1.0.0\publishing`

Откройте **Windows PowerShell**. Проверьте инструменты:

```powershell
git --version
gh --version
py -3.11 --version
```

Если `gh` отсутствует, установите его, затем откройте новое окно PowerShell:

```powershell
winget install --id GitHub.cli --exact --source winget
```

Если отсутствует Git, его отдельная команда установки:

```powershell
winget install --id Git.Git --exact --source winget
```

Команды ниже предназначены для Windows PowerShell 5.1 или PowerShell 7 с установленным Python 3.11. Если пакет находится в другом каталоге, передайте скрипту `-ReleaseRoot` с полным путём к папке, содержащей `repository` и `research-assets`. Из GitHub-клона тот же скрипт доступен в `tools\publishing`, но ему также нужна соседняя папка `research-assets`.

Теперь войдите именно в аккаунт **KuznetsovKarazin**:

```powershell
gh auth login --hostname github.com --git-protocol https --web --scopes workflow
gh api user --jq '.login'
```

Вторая команда должна вывести `KuznetsovKarazin`. Scope `workflow` нужен для включённого файла `.github/workflows/verify.yml`. Вход через браузер описан в [GitHub CLI](https://cli.github.com/manual/gh_auth_login). Если вы уже авторизованы, но прежнему входу не хватает этого scope, выполните:

```powershell
gh auth refresh --hostname github.com --scopes workflow
```

Запустите подготовленный скрипт:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File 'E:\AURA\AURA-KSP_PUBLIC_RELEASE_v1.0.0\publishing\publish_github.ps1'
```

`ExecutionPolicy Bypass` действует только для этого запуска PowerShell. Скрипт сначала проверит все SHA-256, создаст два архива и sidecar, проверит аккаунт, загрузит код в `KuznetsovKarazin/aura-ksp` и создаст релиз `v1.0.0`. До публикации релиза он скачает загруженные файлы обратно и проверит их SHA-256. Git history и существующие теги не переписываются; при несовпадении он останавливается.

Успешный результат:

```text
GITHUB_RELEASE_PUBLISHED
https://github.com/KuznetsovKarazin/aura-ksp/releases/tag/v1.0.0
```

Результат с реальными commit и URL сохранится в `publish-files\PUBLISH_RESULT.json`. Отдельно откройте **Actions** на GitHub и проверьте завершение CI. Скрипт не сообщает об успешном CI без фактического результата.

Если загрузка прервалась, повторите **ту же команду**: скрипт проверяет уже существующие файлы и не заменяет их. Если он сообщает, что репозиторий уже содержит другую историю или тег указывает на другой commit, пришлите это сообщение; повторять с `--force` не нужно. Официальные команды: [создание репозитория](https://cli.github.com/manual/gh_repo_create), [создание релиза](https://cli.github.com/manual/gh_release_create), [загрузка release assets](https://cli.github.com/manual/gh_release_upload).

При необходимости доступен режим только локальной проверки и упаковки, без обращения к GitHub:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File 'E:\AURA\AURA-KSP_PUBLIC_RELEASE_v1.0.0\publishing\publish_github.ps1' -PrepareOnly
```

## Zenodo: одна запись с кодом и результатами

Для этого проекта проще создать запись вручную, сохранив код, веса и результаты вместе. Не включайте параллельно автоматическую GitHub-интеграцию для `aura-ksp`: она создаёт свой объект из релиза, а нам нужна контролируемая запись с обоими архивами. [Zenodo: GitHub and Software](https://help.zenodo.org/docs/github/).

1. Войдите в [Zenodo](https://zenodo.org/) и создайте **New upload**.
2. Тип ресурса — **Software**; версия — `1.0.0`.
3. В DOI выберите, что существующего DOI нет, затем **Get a DOI now!**. Зарезервированный DOI можно использовать в документах; публично он начнёт работать после публикации. [Инструкция Zenodo](https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/).
4. Загрузите из `E:\AURA\AURA-KSP_PUBLIC_RELEASE_v1.0.0\publish-files` ровно эти четыре файла:

   - `AURA-KSP_repository_v1.0.0.zip`
   - `AURA-KSP_repository_v1.0.0.zip.sha256`
   - `AURA-KSP_research-assets_v1.0.0.zip`
   - `AURA-KSP_research-assets_v1.0.0.zip.sha256`

5. Название: **AURA-KSP: Auditable Stream Reduction for Skeleton Action Recognition**. Авторы, ORCID и аффилиации уже приведены в `repository\CITATION.cff`; сохраните тот же порядок шести авторов. В описании можно использовать `ZENODO_DESCRIPTION.html`. Лицензия — **Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)**. Она согласована с `repository\LICENSE`; границы для сторонних материалов пояснены в `LICENSE_SCOPE.md`. Исходный NTU RGB+D в загрузку не входит.
6. В связанную ссылку добавьте `https://github.com/KuznetsovKarazin/aura-ksp/releases/tag/v1.0.0`. О финансировании укажите согласованную формулировку: **This research has been funded by the Science Committee of the Ministry of Science and Higher Education of the Republic of Kazakhstan (Grant No. AP23486538 Research and development of a system for recognizing images in video streams based on artificial intelligence).** Номер проекта: **AP23486538**. Эти реквизиты уже внесены в `zenodo_metadata.json` и описание записи.
7. Проверьте четыре файла, версию, авторов и лицензию, затем нажмите **Publish**.

Скачайте оба ZIP из уже опубликованной записи в отдельную папку и сравните их SHA-256 с локальными `.sha256`:

```powershell
Get-FileHash -LiteralPath 'ПОЛНЫЙ_ПУТЬ_К_СКАЧАННОМУ_ZIP' -Algorithm SHA256
```

Пришлите публичный DOI Zenodo. Его можно добавить в статью и текущий CFF обычным новым commit. Уже опубликованный тег `v1.0.0` и его файлы при этом сохраняются: артефакт не обязан содержать собственный DOI внутри исходного ZIP. В статье следует цитировать DOI конкретной версии.

Сами эти инструкции и скрипты ничего не публикуют до запуска пользователем. PowerShell-сценарий подготовлен по официальному синтаксису CLI; его проверка в Windows и авторизованные операции выполняются при запуске на вашем ПК.

Статья и её авторские рисунки отдельно доступны под CC BY 4.0; CC BY-NC 4.0 относится к исследовательскому коду и активам. Это отражено в LICENSE_SCOPE.md.
