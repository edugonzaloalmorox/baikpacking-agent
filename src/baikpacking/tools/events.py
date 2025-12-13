from typing import Dict, List



EVENT_KEYWORDS = [
    "303 lucerne",
    "accursed race",
    "accursed race no2",
    "across andes",
    "across andes patagonia verde",
    "alps divide",
    "amersfoort-sauerland-amersfoort",
    "andean raid",
    "ardennes monster",
    "audax gravel",
    "audax road",
    "audax trail",
    "b-hard",
    "b-hard ultra race and brevet",
    "basajaun",
    "bee line 200",
    "berlin munich berlin",
    "bike of the tour divide",
    "bike of the tour divide dotwatcher team edition",
    "borderland 500",
    "bright midnight",
    "capitals",
    "dead ends & cake",
    "dead ends & dolci",
    "dead ends and cake",
    "dales divide",
    "doom",
    "elevation vercors",
    "further elements",
    "further perseverance",
    "further perseverance pyrenees",
    "further pyrenees le chemin",
    "gbduro",
    "gbduro22",
    "gbduro23",
    "gbduro24",
    "gbduro25",
    "granguanche audax gravel",
    "granguanche audax road",
    "granguanche audax trail",
    "granguanche trail",
    "great british divide",
    "great british escapades",
    "gravel birds",
    "gravel del fuego",
    "hamburg's backyard",
    "hardennes gravel tour",
    "headstock 200",
    "headstock 500",
    "highland trail 550",
    "hope 1000",
    "istra land",
    "journey around rwanda",
    "kromvojoj",
    "lakes 'n' knodel",
    "lakes n knodel",
    "lakes ‘n’ knödel",
    "le pilgrimage",
    "le tour de frankie",
    "liege-paris-liege",
    "log drivers waltz",
    "log driver's waltz",
    "madrid to barcelona",
    "memory bike adventure",
    "mittelgebirge classique",
    "mittelgebirgeclassique",
    "mother north",
    "nordic chase",
    "norfolk 360",
    "pan celtic race",
    "peaks and plains",
    "pedalma madrid to barcelona",
    "peninsular divide",
    "perfidious albion",
    "pirenaica",
    "poco loco",
    "pure peak grit",
    "race around rwanda",
    "race around the netherlands",
    "race around the netherlands gx",
    "seven serpents",
    "seven serpents illyrian loop",
    "seven serpents quick bite",
    "seven serpents quick bite!",
    "sneak peaks",
    "solstice sprint",
    "southern divide",
    "southern divide - autumn edition",
    "southern divide - spring edition",
    "super brevet berlin munich berlin",
    "supergrevet munich milan",
    "supergrevet vienna berlin",
    "such24",
    "taunus bikepacking",
    "taunus bikepacking no.5",
    "taunus bikepacking no.6",
    "taunus bikepacking no.7",
    "taunus bikepacking no.8",
    "the accursed race",
    "the alps divide",
    "the bike of the touriste routier",
    "the bright midnight",
    "the capitals",
    "... the capitals 2024",
    "the great british divide",
    "the hills have bikes",
    "the land between",
    "the wild west country",
    "three peaks bike race",
    "three peaks bike race 2023",
    "three peaks bike race 2025",
    "tour de farce",
    "trans balkan race",
    "trans balkans",
    "trans balkans race",
    "trans pyrenees race",
    "transatlantic way",
    "transcontinental",
    "transcontinental race no10",
    "transcontinental race no11",
    "transiberica",
    "transiberica 2023",
    "transiberica 2024",
    "transpyrenees (transiberica)",
    "transpyrenees by transiberica",
    "tcr",
    "the southern divide",
    "the unknown race",
    "tour te waipounamu",
    "touriste routier",
    "two volcano sprint",
    "two volcano sprint 2020",
    "two volcano sprint 2021",
    "two volcano sprint 2024",
    "utrecht ultra",
    "utrecht ultra xl",
    "via race",
    "victoria divide",
    "wild west country",
]

EVENT_ALIASES: Dict[str, List[str]] = {

    # ----------------------- GranGuanche -----------------------
    "granguanche-audax-trail": [
        "granguanche audax trail", "granguanche trail", "audax trail"
    ],
    "granguanche-audax-gravel": [
        "granguanche audax gravel", "audax gravel"
    ],
    "granguanche-audax-road": [
        "granguanche audax road", "audax road"
    ],

    # ----------------------- Further ---------------------------
    "further-elements": ["further elements"],
    "further-perseverance": ["further perseverance"],
    "further-perseverance-pyrenees": ["further perseverance pyrenees"],
    "further-pyrenees-le-chemin": ["further pyrenees le chemin"],

    # ----------------------- Peninsular Divide -----------------
    "peninsular-divide": ["peninsular divide"],

    # ----------------------- Tour de Farce ----------------------
    "tour-de-farce": ["tour de farce"],

    # ----------------------- Trans Pyrenees ---------------------
    "trans-pyrenees-race": ["trans pyrenees race", "transpyrenees"],
    "transpyrenees-transiberica": ["transpyrenees by transiberica", "transpyrenees (transiberica)"],

    # ----------------------- Pirenaica --------------------------
    "pirenaica": ["pirenaica"],

    # ----------------------- Istra Land -------------------------
    "istra-land": ["istra land"],

    # ----------------------- Bohemian Border Bash ---------------
    "bohemian-border-bash-race": ["bohemian border bash race"],

    # ----------------------- Lakes 'n' Knödel -------------------
    "lakes-n-knodel": ["lakes 'n' knodel", "lakes n knodel", "lakes ‘n’ knödel"],

    # ----------------------- Sneak Peaks ------------------------
    "sneak-peaks": ["sneak peaks"],

    # ----------------------- Supergrevet ------------------------
    "supergrevet-vienna-berlin": ["supergrevet vienna berlin"],
    "supergrevet-berlin-munich-berlin": ["super brevet berlin munich berlin"],
    "supergrevet-munich-milan": ["supergrevet munich milan"],

    # ----------------------- Log Drivers Waltz ------------------
    "log-drivers-waltz": ["log drivers waltz", "log driver's waltz"],

    # ----------------------- The Land Between -------------------
    "the-land-between": ["the land between"],

    # ----------------------- Ardennes Monster -------------------
    "ardennes-monster": ["ardennes monster"],

    # ----------------------- Bentang Jawa -----------------------
    "bentang-jawa": ["bentang jawa"],

    # ----------------------- GBDURO -----------------------------
    "gbduro": ["gbduro", "gbduro24", "gbduro25", "gbduro23", "gbduro22"],

    # ----------------------- Berlin–Munich–Berlin ---------------
    "berlin-munich-berlin": ["berlin munich berlin"],

    # ----------------------- Basajaun ---------------------------
    "basajaun": ["basajaun"],

    # ----------------------- VIA Race ---------------------------
    "via-race": ["via race"],

    # ----------------------- Transcontinental -------------------
    "transcontinental": ["transcontinental", "tcr", "transcontinental race no10", "transcontinental race no11"],

    # ----------------------- Hills Have Bikes -------------------
    "hills-have-bikes": ["the hills have bikes"],

    # ----------------------- Utrecht Ultra ----------------------
    "utrecht-ultra": ["utrecht ultra", "utrecht ultra xl"],

    # ----------------------- Capitals by Pedalma ----------------
    "capitals-by-pedalma": [
        "the capitals by pedalma",
        "the capitals",
        "capitals",
        "... the capitals 2024"
    ],

    # ----------------------- Three Peaks Bike Race --------------
    "three-peaks-bike-race": [
        "three peaks bike race",
        "three peaks bike race 2023",
        "three peaks bike race 2025"
    ],

    # ----------------------- Bright Midnight --------------------
    "bright-midnight": ["bright midnight", "the bright midnight"],

    # ----------------------- Peak Grit --------------------------
    "pure-peak-grit": ["pure peak grit"],

    # ----------------------- Andean Raid ------------------------
    "andean-raid": ["andean raid"],

    # ----------------------- Dead Ends & Cake -------------------
    "dead-ends-and-cake": ["dead ends and cake", "dead ends & dolci", "dead ends & cake"],

    # ----------------------- Solstice Sprint --------------------
    "solstice-sprint": ["solstice sprint"],

    # ----------------------- Bike of the Tour Divide ------------
    "bike-of-tour-divide": [
        "bike of the tour divide",
        "bike of the tour divide dotwatcher team edition"
    ],

    # ----------------------- Taunus Bikepacking -----------------
    "taunus-bikepacking": [
        "taunus bikepacking",
        "taunus bikepacking no.7",
        "taunus bikepacking no.8",
        "taunus bikepacking no.6",
        "taunus bikepacking no.5"
    ],

    # ----------------------- Touriste Routier -------------------
    "touriste-routier": ["the bike of the touriste routier"],

    # ----------------------- Race Around The Netherlands --------
    "race-around-netherlands": [
        "race around the netherlands",
        "race around the netherlands gx",
    ],

    # ----------------------- Nordic Chase -----------------------
    "nordic-chase": ["nordic chase"],

    # ----------------------- Hamburg’s Backyard -----------------
    "hamburgs-backyard": ["hamburg's backyard"],

    # ----------------------- Mittelgebirge Classique ------------
    "mittelgebirge-classique": [
        "mittelgebirge classique",
        "mittelgebirgeclassique"
    ],

    # ----------------------- Amersfoort-Sauerland ---------------
    "amersfoort-sauerland-amersfoort": [
        "amersfoort-sauerland-amersfoort"
    ],

    # ----------------------- Trans Balkan ------------------------
    "trans-balkan-race": [
        "trans balkan race",
        "trans balkans race",
        "trans balkans"
    ],

    # ----------------------- Great British Escapades ------------
    "great-british-escapades": ["great british escapades"],

    # ----------------------- Hardennes Gravel Tour --------------
    "hardennes-gravel-tour": ["hardennes gravel tour"],

    # ----------------------- Pedalma M2B -------------------------
    "madrid-to-barcelona": [
        "pedalma madrid to barcelona",
        "madrid to barcelona"
    ],

    # ----------------------- Headstock --------------------------
    "headstock-500": ["headstock 500"],
    "headstock-200": ["headstock 200"],

    # ----------------------- Highland Trail ---------------------
    "highland-trail-550": ["highland trail 550", "the highland trail 550"],

    # ----------------------- Peaks and Plains -------------------
    "peaks-and-plains": ["peaks and plains"],

    # ----------------------- Seven Serpents ----------------------
    "seven-serpents": [
        "seven serpents",
        "seven serpents quick bite",
        "seven serpents quick bite!",
        "seven serpents illyrian loop"
    ],

    # ----------------------- Bee Line 200 ------------------------
    "bee-line-200": ["bee line 200"],

    # ----------------------- 303 Lucerne -------------------------
    "303-lucerne": ["303 lucerne"],

    # ----------------------- The Accursed Race ------------------
    "accursed-race": [
        "the accursed race",
        "the accursed race no2"
    ],

    # ----------------------- Wild West Country -------------------
    "wild-west-country": ["the wild west country", "wild west country"],

    # ----------------------- Southern Divide ---------------------
    "southern-divide": [
        "the southern divide", 
        "the southern divide - spring edition",
        "the southern divide - autumn edition"
    ],

    # ----------------------- Gravel Birds -----------------------
    "gravel-birds": ["gravel birds"],

    # ----------------------- Dales Divide ------------------------
    "dales-divide": ["dales divide"],

    # ----------------------- Le Tour de Frankie -----------------
    "le-tour-de-frankie": ["le tour de frankie"],

    # ----------------------- Unknown Race ------------------------
    "unknown-race": ["the unknown race"],

    # ----------------------- Norfolk 360 -------------------------
    "norfolk-360": ["norfolk 360"],

    # ----------------------- Doom -------------------------------
    "doom": ["doom"],

    # ----------------------- Race Around Rwanda -----------------
    "race-around-rwanda": ["race around rwanda"],

    # ----------------------- Across Andes ------------------------
    "across-andes": ["across andes", "across andes patagonia verde"],

    # ----------------------- Two Volcano Sprint ------------------
    "two-volcano-sprint": [
        "two volcano sprint",
        "two volcano sprint 2021",
        "two volcano sprint 2024",
        "two volcano sprint 2020"
    ],

    # ----------------------- Borderland 500 ----------------------
    "borderland-500": ["borderland 500"],

    # ----------------------- Le Pilgrimage ------------------------
    "le-pilgrimage": ["le pilgrimage"],

    # ----------------------- SUCH24 ------------------------------
    "such24": ["such24"],

    # ----------------------- Alps Divide -------------------------
    "alps-divide": ["alps divide", "the alps divide"],

    # ----------------------- TransIberica -------------------------
    "transiberica": ["transiberica", "transibérica", "transiberica 2023", "transiberica 2024"],

    # ----------------------- Liege–Paris–Liege -------------------
    "liege-paris-liege": ["liège-paris-liège", "liege-paris-liege"],

    # ----------------------- Perfidious Albion -------------------
    "perfidious-albion": ["the perfidious albion"],

    # ----------------------- Great British Divide ----------------
    "great-british-divide": [
        "great british divide",
        "the great british divide"
    ],

    # ----------------------- Colorado Trail Race -----------------
    "colorado-trail-race": ["colorado trail race"],

    # ----------------------- Mother North ------------------------
    "mother-north": ["mother north"],

    # ----------------------- TransAtlantic Way -------------------
    "transatlantic-way": ["transatlantic way", "the transatlantic way"],

    # ----------------------- Pan Celtic --------------------------
    "pan-celtic-race": ["pan celtic race"],

    # ----------------------- Elevation Vercors -------------------
    "elevation-vercors": ["elevation vercors"],

    # ----------------------- Memory Bike Adventure ---------------
    "memory-bike-adventure": ["memory bike adventure"],

    # ----------------------- Hope 1000 ---------------------------
    "hope-1000": ["hope 1000"],

    # ----------------------- Blaenau 600 -------------------------
    "blaenau-600": ["blaenau 600"],

    # ----------------------- B-HARD Ultra ------------------------
    "bhard-ultra": ["b-hard ultra race and brevet", "b-hard"],

    # ----------------------- Kromvojoj ---------------------------
    "kromvojoj": ["kromvojoj"],

    # ----------------------- Unmapping Sweden --------------------
    "unmapping-sweden": ["unmapping sweden", "unmapping: sweden"],

    # ----------------------- Gravel del Fuego --------------------
    "gravel-del-fuego": ["gravel del fuego"],

    # ----------------------- Poco Loco ---------------------------
    "poco-loco": ["poco loco"],

    # ----------------------- Journey Around Rwanda ---------------
    "journey-around-rwanda": ["journey around rwanda"],

    # ----------------------- Victoria Divide ---------------------
    "victoria-divide": ["victoria divide"],
}