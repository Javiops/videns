"""Nombres de campeón: la playerlist del cliente usa nombres de display y
Match-V5/ddragon usan el interno. Mapa de los que difieren (ampliar cuando la
QA de una partida nueva delate uno que falte — process_game avisa)."""

DISPLAY_TO_INTERNAL = {
    "Cho'Gath": "Chogath",
    "Wukong": "MonkeyKing",
    "Kai'Sa": "Kaisa",
    "Kha'Zix": "Khazix",
    "Vel'Koz": "Velkoz",
    "Rek'Sai": "RekSai",
    "K'Sante": "KSante",
    "Bel'Veth": "Belveth",
    "Nunu & Willump": "Nunu",
    "Renata Glasc": "Renata",
    "Dr. Mundo": "DrMundo",
    "Jarvan IV": "JarvanIV",
    "Lee Sin": "LeeSin",
    "Master Yi": "MasterYi",
    "Miss Fortune": "MissFortune",
    "Twisted Fate": "TwistedFate",
    "Xin Zhao": "XinZhao",
    "Aurelion Sol": "AurelionSol",
    "Tahm Kench": "TahmKench",
    "LeBlanc": "Leblanc",
    "Fiddlesticks": "FiddleSticks",
}


def internal(name: str) -> str:
    return DISPLAY_TO_INTERNAL.get(name, name)
