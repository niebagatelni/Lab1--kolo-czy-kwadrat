# Koło czy kwadrat

Program uczy się odróżniać koło od kwadratu i rozpoznaje figury narysowane myszką.

Uruchomienie:

```
python app.py
```

Wymagane biblioteki: numpy, Pillow, matplotlib (tkinter jest w standardowym Pythonie).

## Co robi program

W oknie są trzy zakładki:

| Zakładka | Do czego służy |
|---|---|
| **Ucz** | Komputer sam rysuje tysiące kół i kwadratów (przycisk „Generuj", dopisuje do `data.csv`) i uczy model. Można też narysować figurę i oznaczyć ją przyciskiem „To koło" / „To kwadrat". |
| **Rozpoznaj** | Rysujesz figurę jednym pociągnięciem (obrys sam się domyka). Program mówi „koło" albo „kwadrat" i pokazuje na wykresach, gdzie Twój rysunek wypadł. |
| **Symulacja** | Animacja tego, jak model uczył się na danych: kolejne punkty na wykresach i przesuwająca się linia granicy. |

## Co robią poszczególne pliki

W folderze jest 7 plików z kodem. Uruchamiasz tylko `app.py`, a reszta to jego pomocnicy.

| Plik | Co robi |
|---|---|
| `app.py` | Okno programu z trzema zakładkami. Obsługuje myszkę (rysowanie jednym pociągnięciem, domykanie figury) i przyciski. Sam niczego nie liczy, tylko woła pozostałe pliki i pokazuje wyniki. |
| `feature_extractor.py` | Zamienia narysowaną linię na 4 liczby (cechy). Używają go wszyscy, więc każda figura jest zawsze oceniana tak samo. |
| `data_generator.py` | Komputer sam „rysuje" w pamięci tysiące kół i kwadratów (z drżeniem ręki, owalnością, niedomkniętą linią) i od razu liczy ich cechy. Obrazków nie zapisuje. |
| `data_handler.py` | Dopisuje wiersze do `data.csv` i je odczytuje. Niczego nie kasuje. Pilnuje też, żeby nie zmieszać danych z różnych wersji cech. |
| `model.py` | Uczy się na danych: nadaje wagi cechom, krok po kroku. Wynik zapisuje w `boundary.json`, a potem na jego podstawie mówi „koło" albo „kwadrat" z pewnością w procentach. |
| `simulation.py` | Animacja uczenia: 4 wykresy cech z kolejno pojawiającymi się punktami i przesuwającą się linią granicy oraz wykres z wagami. |
| `recognition_chart.py` | Wykresy w zakładce *Rozpoznaj*: gdzie wypadł Twój rysunek (gwiazdka) względem danych uczących i która cecha głosowała za kołem, a która za kwadratem. |

Pliki z danymi, które program tworzy:

| Plik | Zawartość |
|---|---|
| `data.csv` | Tabela: jeden wiersz to jedna figura (4 cechy + etykieta + źródło). Dane tylko się dopisuje. |
| `boundary.json` | To, czego model się nauczył: wagi cech i stała. |
| `simulation.png` | Ostatnia klatka animacji. |

Przepływ danych:

```
generator (lub Twój rysunek) -> ekstrakcja cech -> data.csv -> uczenie -> boundary.json -> rozpoznawanie
```

## Na czym polegają cechy

Program nie „widzi" koła ani kwadratu. Z narysowanej linii wyciąga **4 liczby** opisujące kształt, a decyzję podejmuje dopiero na ich podstawie. Wszystkie są względne, więc nie zależą od wielkości ani obrotu figury.

### 1. Falowanie promienia

Program mierzy odległość każdego punktu linii od środka figury i sprawdza, jak bardzo te odległości się różnią.

- **Koło:** wszystkie punkty są mniej więcej tak samo daleko, więc falowanie jest małe.
- **Kwadrat:** rogi są dalej od środka niż środki boków, więc falowanie jest duże.

Przed pomiarem program „prostuje elipsę": rozciąga rysunek tak, żeby owal stał się kołem. Dzięki temu owalne koło nie wygląda jak kwadrat. Kwadrat po takim zabiegu zostaje kwadratem.

### 2. Rozpiętość promienia

To ta sama idea, tylko inaczej policzona: różnica między prawie najdalszym a prawie najbliższym punktem (5% skrajnych punktów jest pomijane), w stosunku do tego najdalszego. Jest mniej wrażliwa na pojedynczą krzywą kreskę. Obie cechy promienia mówią więc prawie to samo.

### 3. Kolistość

Wyobraź sobie gumkę naciągniętą wokół figury. Program porównuje pole tej gumki z jej obwodem.

- **Idealne koło** daje ok. 1,0.
- **Kwadrat** daje ok. 0,78–0,88 (narysowany odręcznie).
- **Owal** wychodzi pośrodku.

Gumka „nie widzi" wgłębień, więc drżenie ręki prawie nie zmienia tej cechy. Dlatego jest jedną z najpewniejszych.

### 4. Liczba rogów

Program upraszcza tę gumkę do wielokąta o jak najmniejszej liczbie wierzchołków, tak żeby kształt prawie się nie zmienił, i liczy wierzchołki.

- **Kwadrat:** 4 (odręcznie 4–5).
- **Koło:** ok. 8. To nie są prawdziwe rogi, tylko tyle odcinków potrzeba, żeby przybliżyć okrąg przy tej tolerancji.

Zaokrąglone rogi kwadratu zwiększają tę liczbę, więc cecha jest pewna dla ostrych kwadratów, a mniej dla bardzo zaokrąglonych.

### Typowe wartości (z 20 narysowanych odręcznie figur)

| Cecha | Koła | Kwadraty |
|---|---|---|
| Falowanie promienia | 0,04–0,13 | 0,10–0,15 |
| Rozpiętość promienia | 0,12–0,37 | 0,27–0,37 |
| Kolistość | 0,91–0,99 | 0,79–0,88 |
| Liczba rogów | 7–8 | 4–5 |

Cechy promienia **zachodzą na siebie**, bo niechlujne koło też faluje. Dlatego rozstrzygają głównie **kolistość i liczba rogów**, które wyraźnie rozdzielają obie figury.

## Jak model łączy cechy

Każda cecha ma wagę. Model sumuje: *waga × cecha* dla wszystkich czterech, plus stałą. Wynik dodatni oznacza kwadrat, ujemny koło. Waga mówi, jak bardzo model ufa danej cesze.

Na wykresach w zakładce *Rozpoznaj* widać ten rachunek: słupki to wpływ każdej cechy, a ostatni słupek to suma.
