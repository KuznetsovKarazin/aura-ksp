# Конкретная проверка лицензий AURA-KSP — 20 сентября 2026

**Статус выпуска:** нижеследующий отчёт сохраняет формулировки первоначальной проверки. В релизе 1.0.0 её кандидат уже оформлен как LICENSE/THIRD_PARTY_NOTICES; актуальная область действия, включая CC BY 4.0 для статьи, определена в LICENSE_SCOPE.md.

## Практический вывод

Для проверенного кода разумный согласованный вариант — **CC BY-NC 4.0**, с сохранением лицензии CTR-GCN, атрибуции, указанием адаптаций и областью действия. Это устраняет обнаруженный пробел в атрибуции; менять архитектуру, переписывать подтверждённый runtime или выдумывать «чистую» независимую реализацию не требуется. Лицензия всего проекта должна быть принята правообладателями; каталог `candidate/` содержит готовый вариант для такого решения, а не утверждение, что решение уже принято.

Код/статью/агрегированные результаты можно оформить этим способом. Для весов и пообъектных предсказаний предлагается scope «результаты исследовательского эксперимента; лицензируются только права авторов на них; исходные права NTU сохраняются». **CC BY-NC сама по себе не предоставляет разрешение владельца NTU** и не исправляет нарушение условий доступа. Не заявляем найденное специальное разрешение на checkpoints или logits: его на открытой странице нет. Это конкретная оставшаяся граница условий, а не нерешённость лицензирования самого кода.

## Что именно сопоставлено

Получены все 24 Python-файла официального CTR-GCN на commit `e13d7582e281d06711eeecb380a472b278ae1663` и сравнены с 57 Python-файлами подготовленного repository. `TOKEN_COMPARISON.json` фиксирует SHA-256, точные совпадающие токен-блоки и строки. Игнорировались пробелы/комментарии, порог — 35 последовательных токенов; это вспомогательный поиск, не доказательство самостоятельности остальных файлов. Значимые функции проверены также по структуре и операциям.

- `models/ctrgcn_reference.py`: явная структурная адаптация `model/ctrgcn.py`: четыре проекции CTRGC, усреднение по времени, tanh разности отношений, alpha и adjacency, einsum; три spatial subsets, маленькая BN initialization; temporal branches; schedule 64×4,128×3,256×3 со strides в блоках 5 и 8; одинаковая перестановка/нормализация входа. Точные блоки включают local 47–51 ↔ upstream 55–59 и 113–117 ↔ 114–118. Локально добавлены типы, проверки, nn.Identity/Zero, buffer, feature API и другая организация классов.
- `models/ctrgcn.py`: переименованные операторы q/k/v/relation, но та же CTRGC конструкция и block schedule. Это упрощённая предыдущая реализация, а не основание объявить файл независимым от CTR-GCN. В notice она также отнесена к адаптациям.
- `data/reference.py`: `valid_crop_bounds` и `valid_crop_resize` соответствуют official `feeders/tools.py:valid_crop_resize`: центрированный crop либо случайная доля с минимумом 64; преобразование C,T,V,M → C·V·M,T; bilinear interpolation с align_corners=False; возврат к исходному расположению осей. Код переработан, добавлены Generator, validation и selected indices. NTU bone pairs имеют точное длинное совпадение с `feeders/bone_pairs.py`.
- `models/graph.py`: 24 inward NTU edges совпадают с `graph/ntu_rgb_d.py`; identity/inward/outward subsets соответствуют `graph/tools.py`. Нормализация переписана векторно. Граф NTU — также фактическая спецификация benchmark; notice не объявляет исключительные права на неё.
- `data/ntu.py` и `data/ntu120.py`: найденные длинные совпадения — списки официальных train subjects, а не скопированный raw parser. Локальный парсер сохраняет координатные треки и ранжирует длину+движение; upstream SGN parser собирает joints/colors/interval и удаляет empty frames, затем применяет отдельный denoising. Это различающиеся реализации. Ни `get_raw_skes_data.py`, ни `get_raw_denoised_data.py`, ни `seq_transformation.py` не включены в готовый runtime под прежними именами.
- Остальные Python-файлы не дали ≥35-token совпадения с pinned CTR-GCN. Это не юридический сертификат происхождения; важнее, что обнаруженные архитектурные соответствия, в том числе переименованные, явно атрибутированы. Runtime imports используют библиотеки как зависимости, их код не перелицензируется.

CTR-GCN прямо называет 2s-AGCN, SGN и HCN. Поэтому эти credits сохранены. SGN-файлы upstream CTR-GCN действительно имеют Microsoft/MIT headers. Нельзя подменять этот MIT copyright общей CC-лицензией. Однако их деревья не скопированы в распространяемый AURA runtime; наличие HCN в upstream acknowledgements само по себе не создаёт обязательства получить отдельную лицензию HCN на весь AURA-код. Дополнительные загруженные исходники служат проверке; включать их в публичный архив целиком не требуется.


Дополнительно напрямую загружены официальные repositories 2s-AGCN (`953c14fc10883cd869646328f5d522e9e9282063`, 22 файлов), SGN (`79aa0dd89e332c3c14bb70a575905d35cbcdbd86`, 10) и HCN (`98b15c5c6128ba843231486f5e040728d265c1a7`, 13). Семь локальных model/data файлов сопоставлены со всеми Python-файлами этих источников; `LINEAGE_TOKEN_COMPARISON.json` показывает только benchmark split constants и bone/graph lists выше порога. Визуальная проверка HCN `read_skeleton/read_xyz` также обнаружила иную реализацию: словари frameInfo/bodyInfo/jointInfo и первые max_body вместо локального body-ID tracking/ranking. Проверка не нашла конкретного скопированного HCN parser, для которого потребовалась бы неизвестная HCN лицензия. SGN MIT и 2s-AGCN CC BY-NC license сохранены побайтово в candidate/UPSTREAM_LICENSES, а проверенные commits/URLs/sha — в provenance JSON. Это позволяет закрыть прежний общий «HCN licensing gap» для проверенного выпуска; невозможность доказать отсутствие любого заимствования не превращается в автоматический запрет на релиз.

## Состав данных и область NTU

В `research-assets` насчитано 299 файлов: 272 JSON, 8 CSV, 6 JSONL, 5 NPZ, 4 PT, 2 GZ, TXT и MD. Исходных skeleton/video/coordinate-cache файлов в этом перечне нет. Во всех пяти NPZ массивы открыты с allow_pickle=False: четыре содержат logits 59477×120, labels, setups, sample IDs и selected frame indices; paired NPZ содержит logits, decisions, confusion matrices и bootstrap 10000×3. Координат суставов нет. `ASSET_ARRAY_INVENTORY.json` фиксирует схему. Четыре checkpoint имеют контейнер PyTorch с параметрами модели и config/binding metadata; текущая проверка прочитала pickle opcodes без исполнения, детальный tensor-аудит выполнен ранее в проекте. JSONL сохраняют идентификаторы и относительные имена исходных samples, а не байты соответствующих skeletons.

Официальные [условия NTU](https://rose1.ntu.edu.sg/dataset/actionRecognition/) разрешают некоммерческие академические исследования, требуют оформления доступа; без разрешения ограничивают распространение и создание новых датасетов. Для изображений действуют отдельные ограничения демонстрации. На этой странице не выделены trained weights и numerical predictions. Поэтому нельзя честно написать «NTU явно разрешает любые checkpoint/logits», но и нет основания выдавать вымышленный прямой запрет на публикацию весов за цитату из условий. Проверенный пакет результатов не содержит движений и не заменяет benchmark. Рекомендация публиковать его как supplementary research outputs является интерпретацией состава и условий, а не разрешением ROSE Lab. Назвать файлы outputs недостаточно, если их фактическое содержание превращается в новый dataset; здесь координат нет. Оригинальные данные/кеши Stage 1/frames.npy исключить. Если нужен бесспорный ответ именно о публичном per-sample пакете, адресный вопрос владельцу должен перечислять только веса, logits, labels/IDs, а не просить разрешение на весь проект.

## Точные изменения для релиза

1. Принять выбранную авторами лицензию и поместить `candidate/LICENSE`, `LICENSE_SCOPE.md`, `THIRD_PARTY_NOTICES.md` в корень нового release.
2. Скопировать `candidate/UPSTREAM_LICENSES/` и provenance JSON. Сохранить CTR-GCN license неизменённым. Не дописывать copyright-author имена, которые upstream сам не заявил; авторы paper указаны как attribution.
3. В README/CITATION/Zenodo согласовать CC-BY-NC-4.0 и scope. Не ставить MIT/Apache поверх адаптированного CTR-GCN. Не обозначать NTU как CC-licensed dataset AURA.
4. Не менять исторические source pins, manifest или байты runtime ради headers: корневой notice даёт path mapping и change statement. Новый релиз получает собственный manifest поверх сохранённых внутренних pins.
5. Не включать raw skeleton/RGB/depth/IR, frames.npy и координатные NPZ. Оставить ссылки на официальный механизм доступа. Сохранить две dataset citations в статье.

## Первичные источники

- CTR-GCN source and license: https://github.com/Uason-Chen/CTR-GCN/tree/e13d7582e281d06711eeecb380a472b278ae1663
- CTR-GCN license: https://github.com/Uason-Chen/CTR-GCN/blob/e13d7582e281d06711eeecb380a472b278ae1663/LICENSE
- 2s-AGCN license: https://github.com/lshiwjx/2s-AGCN/blob/master/LICENSE
- SGN license: https://github.com/microsoft/SGN/blob/master/LICENSE
- HCN source: https://github.com/huguyuehuhu/HCN-pytorch
- NTU access/terms: https://rose1.ntu.edu.sg/dataset/actionRecognition/
- Official dataset protocol/naming documentation: https://github.com/shahroudy/NTURGB-D

Это техническое исследование происхождения и условий распространения, без правовой гарантии. Исторические runtime-файлы не изменялись; внешних публикаций не выполнялось.
