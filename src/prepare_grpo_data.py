"""
Prepare language-specific GRPO training data.

Training set: Random word pairs + headlines from HuggingFace datasets
Test set: Word pairs/headlines from TSV files you provide

Usage:
    python prepare_grpo_data.py --language en --test_tsv task-a-en.tsv
    python prepare_grpo_data.py --language zh --test_tsv task-a-zh.tsv
    python prepare_grpo_data.py --language es --test_tsv task-a-es.tsv
"""
import pandas as pd
import random
from datasets import Dataset, load_dataset
import os

# English words for random word pair generation
ENGLISH_NOUNS = [
    # ----- original list -----
    "cat", "dog", "house", "car", "tree", "phone", "book", "chair", "table", "computer",
    "coffee", "pizza", "banana", "apple", "orange", "water", "fire", "ice", "sun", "moon",
    "star", "cloud", "rain", "snow", "wind", "mountain", "river", "ocean", "beach", "forest",
    "bird", "fish", "horse", "cow", "pig", "chicken", "duck", "rabbit", "mouse", "elephant",
    "lion", "tiger", "bear", "monkey", "snake", "frog", "butterfly", "spider", "ant", "bee",
    "doctor", "teacher", "lawyer", "chef", "pilot", "nurse", "artist", "musician", "writer", "engineer",
    "robot", "alien", "zombie", "vampire", "ghost", "wizard", "ninja", "pirate", "knight", "dragon",
    "hammer", "screwdriver", "ladder", "bucket", "rope", "nail", "brush", "mirror", "clock", "lamp",
    "shirt", "pants", "shoes", "hat", "glasses", "watch", "ring", "necklace", "umbrella", "bag",
    "cake", "cookie", "bread", "cheese", "butter", "milk", "egg", "bacon", "salad", "soup",
    "microwave", "refrigerator", "television", "keyboard", "headphones", "camera", "battery", "charger",
    "pencil", "eraser", "scissors", "stapler", "notebook", "envelope", "stamp", "calendar", "map",
    "guitar", "piano", "drum", "violin", "trumpet", "flute", "saxophone", "microphone", "speaker",
    "soccer", "basketball", "tennis", "golf", "baseball", "hockey", "swimming", "skiing", "surfing",
    "airplane", "helicopter", "submarine", "rocket", "spaceship", "bicycle", "motorcycle", "boat", "train",
    # ----- new nouns added: in test file-----
    "flower", "pumpkin", "laptop", "fridge", "pepper", "clothe", "hair", "towel", "corn", "tomato",
    # ----- new nouns added: others-----
    "tea", "sugar", "salt", "rice", "pasta", "beef", "sandwich", "potato", "lettuce", "mushroom", 
    "wrench", "saw", "axe", "desk", "flashlight", "door", "window", "curtain", "bookshelf", 
    "sofa", "couch", "mattress", "pillow", "blanket", "carpet", "rug", "cabinet", "wardrobe",
    "smartphone", "tablet", "monitor", "printer", "scanner", "radio", 
    "plate", "bowl", "cup", "bottle", "glass", "knife", "fork", "spoon",
]

ENGLISH_VERBS = [
    # ----- verbs added: in test file-----
    "hammer", "drill", "spray", "deseed", "roll", "wash", "shake", "measure", "cut", "blend", "move", 
    # ----- verbs added: others-----
    "run", "walk", "jump", "sit", "stand", "lie","sleep", "eat", "drink", "cook", "bake", "boil", 
    "fry", "grill", "stir", "mix", "slice", "peel", "chop", "grind", "pour", "serve", "clean",
    "dry", "fold", "iron", "sew", "knit", "paint", "draw", "write", "read", "type", "click", "scroll", 
    "search", "call", "text", "email", "talk","speak", "listen", "hear", "see", "watch", "look", 
    "smell", "feel", "touch", "saw", "carve", "sand", "polish", "tighten", "loosen",
    "assemble", "disassemble", "build", "repair", "fix", "climb", "dig", "plant", "water", "mow", 
    "sweep", "vacuum", "mop", "dust", "shop","buy", "sell", "pay", "order", "drive", "ride", "park",
    "fly", "sail", "fish", "hunt", "jog", "sprint", "crawl", "slide", "hop", "skip",
    "kneel", "stretch", "steam", "roast", "mash", "whisk", "taste", "shave", "brush", "wear",
    "put", "take", "tie", "zip", "button", "dress", "undress", "like", "love", "prefer", "think", "know",
    "decorate", "arrange", "carry", "lift", "push", "pull", "swing", "spin", "screw", "unscrew", 
    "nail", "plane", "adjust", "install", "upload", "share", "travel", "get", "have", "make", "need", "want",
    "uninstall", "press", "sketch", "photograph", "record", "play", "view", "download",
    "understand", "remember", "forget", "help", "give", "show", "open", "close", "start", "finish",
]


# Chinese words for random word pair generation
CHINESE_NOUNS = [
    "猫", "狗", "房子", "汽车", "树", "电话", "书", "椅子", "桌子", "电脑",
    "咖啡", "披萨", "香蕉", "苹果", "橙子", "水", "火", "冰", "太阳", "月亮",
    "星星", "云", "雨", "雪", "风", "山", "河", "海洋", "海滩", "森林",
    "鸟", "鱼", "马", "牛", "猪", "鸡", "鸭", "兔子", "老鼠", "大象",
    "狮子", "老虎", "熊", "猴子", "蛇", "青蛙", "蝴蝶", "蜘蛛", "蚂蚁", "蜜蜂",
    "医生", "老师", "律师", "厨师", "飞行员", "护士", "艺术家", "音乐家", "作家", "工程师",
    "机器人", "外星人", "僵尸", "吸血鬼", "鬼", "巫师", "忍者", "海盗", "骑士", "龙",
    "锤子", "螺丝刀", "梯子", "桶", "绳子", "钉子", "刷子", "镜子", "时钟", "台灯",
    "衬衫", "裤子", "鞋子", "帽子", "眼镜", "手表", "戒指", "项链", "雨伞", "包",
    "蛋糕", "饼干", "面包", "奶酪", "黄油", "牛奶", "鸡蛋", "培根", "沙拉", "汤",
    "微波炉", "冰箱", "电视", "键盘", "耳机", "相机", "电池", "充电器",
    "铅笔", "橡皮", "剪刀", "订书机", "笔记本", "信封", "邮票", "日历", "地图",
    "吉他", "钢琴", "鼓", "小提琴", "小号", "长笛", "萨克斯管", "麦克风", "扬声器",
    "足球", "篮球", "网球", "高尔夫", "棒球", "曲棍球", "游泳", "滑雪", "冲浪",
    "飞机", "直升机", "潜艇", "火箭", "宇宙飞船", "自行车", "摩托车", "船", "火车",
    # ----- new nouns added: in test file -----
    "花", "南瓜", "笔记本电脑", "冰箱", "胡椒", "衣物", "头发", "毛巾", "玉米", "番茄",
    # ----- new nouns added: others -----
    "茶", "糖", "盐", "米", "意大利面", "牛肉", "三明治", "土豆", "生菜", "蘑菇",
    "扳手", "锯子", "斧头", "书桌", "手电筒", "门", "窗户", "窗帘", "书架",
    "沙发", "长沙发", "床垫", "枕头", "毛毯", "地毯", "小地毯", "橱柜", "衣柜",
    "智能手机", "平板电脑", "显示器", "打印机", "扫描仪", "收音机",
    "盘子", "碗", "杯子", "瓶子", "玻璃杯", "刀", "叉子", "勺子"
]


CHINESE_VERBS = [
    # ----- verbs added: in test file-----
    "锤击", "敲击","喷", "喷洒", "去籽", "滚动", "洗","冲洗", "洗涤", "摇", "摇晃", "测量", 
    "切", "切割","混合", "移动", "钻", "钻井", 
    # ----- verbs added: others-----
    "跑", "走", "跳", "坐", "站", "躺", "睡觉", "吃", "喝", "做饭", "烘烤", "沸腾", 
    "炸", "烧烤", "搅拌", "混合", "切片", "剥皮", "剁碎", "磨碎", "倒", "上菜",
    "打扫", "擦干", "折叠", "熨烫", "缝纫", "编织", "涂漆", "画", "写", "读", "打字",
    "点击", "滚动", "搜索", "打电话", "发短信", "发邮件", "谈话", "说话", "听", "看见", "观看",
    "看", "嗅闻", "感觉", "触摸", "锯", "雕刻", "打磨", "抛光", "收紧", "松开",
    "组装", "拆卸", "建造", "修理", "修复", "爬", "挖", "种植", "浇水", "割草",
    "扫", "吸尘", "拖地", "除尘", "购物", "买", "卖", "付钱", "点餐", "开车", "骑", "停车",
    "飞", "航行", "钓鱼", "狩猎", "慢跑", "冲刺", "爬行", "滑动", "单脚跳", "跳绳",
    "跪下", "伸展", "蒸", "烤", "捣碎", "打发", "品尝", "刮胡子", "刷", "穿",
    "放", "拿", "系", "拉链", "扣", "打扮", "脱衣", "喜欢", "爱", "偏好", "思考", "知道",
    "装饰", "安排", "搬运", "举起", "推", "拉", "摆动", "旋转", "拧螺丝", "松开螺丝",
    "钉", "刨", "调整", "安装", "上传", "分享", "旅行", "得到", "有", "做", "需要", "想要",
    "卸载", "按压", "素描", "拍照", "录音", "玩", "查看", "下载",
    "理解", "记得", "忘记", "帮助", "给", "展示", "打开", "关闭", "开始", "完成"
]


# Spanish words for random word pair generation
SPANISH_NOUNS = [
    "gato", "perro", "casa", "coche", "árbol", "teléfono", "libro", "silla", "mesa", "computadora",
    "café", "pizza", "plátano", "manzana", "naranja", "agua", "fuego", "hielo", "sol", "luna",
    "estrella", "nube", "lluvia", "nieve", "viento", "montaña", "río", "océano", "playa", "bosque",
    "pájaro", "pez", "caballo", "vaca", "cerdo", "pollo", "pato", "conejo", "ratón", "elefante",
    "león", "tigre", "oso", "mono", "serpiente", "rana", "mariposa", "araña", "hormiga", "abeja",
    "médico", "profesor", "abogado", "chef", "piloto", "enfermera", "artista", "músico", "escritor", "ingeniero",
    "robot", "extraterrestre", "zombi", "vampiro", "fantasma", "mago", "ninja", "pirata", "caballero", "dragón",
    "martillo", "destornillador", "escalera", "cubo", "cuerda", "clavo", "cepillo", "espejo", "reloj", "lámpara",
    "camisa", "pantalones", "zapatos", "sombrero", "gafas", "reloj", "anillo", "collar", "paraguas", "bolsa",
    "pastel", "galleta", "pan", "queso", "mantequilla", "leche", "huevo", "tocino", "ensalada", "sopa",
    "microondas", "refrigerador", "televisión", "teclado", "auriculares", "cámara", "batería", "cargador",
    "lápiz", "borrador", "tijeras", "grapadora", "cuaderno", "sobre", "sello", "calendario", "mapa",
    "guitarra", "piano", "tambor", "violín", "trompeta", "flauta", "saxofón", "micrófono", "altavoz",
    "fútbol", "baloncesto", "tenis", "golf", "béisbol", "hockey", "natación", "esquí", "surf",
    "avión", "helicóptero", "submarino", "cohete", "nave espacial", "bicicleta", "motocicleta", "barco", "tren"
    # ----- new nouns added: in test file -----
    "flor", "calabaza", "portátil", "nevera", "pimienta", "ropa", "cabello", "toalla", "maíz", "tomate",
    # ----- new nouns added: others -----
    "té", "azúcar", "sal", "arroz", "pasta", "carne de res", "sándwich", "patata", "lechuga", "champiñón",
    "llave inglesa", "sierra", "hacha", "escritorio", "linterna", "puerta", "ventana", "cortina", "estantería",
    "sofá", "sofá", "colchón", "almohada", "manta", "alfombra", "tapete", "gabinete", "armario",
    "teléfono inteligente", "tableta", "monitor", "impresora", "escáner", "radio",
    "plato", "cuenco", "taza", "botella", "vaso", "cuchillo", "tenedor", "cuchara"
]

SPANISH_VERBS = [
    "martillar", "taladrar", "rociar", "dessembrar", "rodar", "lavar", "sacudir", "medir", "cortar",
    "mezclar", "mover", "correr", "caminar", "saltar", "sentarse", "pararse", "acostarse", "dormir",
    "comer", "beber", "cocinar", "hornear", "hervir", "freír", "asar", "remover", "mezclar",
    "rebanar", "pelar", "picar", "moler", "verter", "servir", "limpiar", "secar", "doblar",
    "planchar", "coser", "tejer", "pintar", "dibujar", "escribir", "leer", "teclear", "clicar",
    "desplazar", "buscar", "llamar", "mensajear", "enviar", "hablar", "hablar", "escuchar",
    "oír", "ver", "observar", "mirar", "oler", "sentir", "tocar", "ver", "tallar", "lijar",
    "pulir", "apretar", "aflojar", "ensamblar", "desensamblar", "construir", "reparar", "arreglar",
    "escalar", "cavar", "plantar", "regar", "barrer", "aspirar", "fregar",
    "desempolvar", "comprar", "comprar", "vender", "pagar", "ordenar", "conducir", "montar",
    "estacionar", "volar", "navegar", "pescar", "cazar", "trotar", "esprintar", "arrastrarse",
    "deslizarse", "brincar", "saltarse", "arrodillarse", "estirar", "vapear", "asar", "triturar",
    "batir", "saborear", "afeitarse", "cepillar", "vestir", "poner", "tomar", "atar",
    "pulsar", "abotonar", "vestirse", "desvestirse", "gustar", "amar", "preferir",
    "pensar", "saber", "decorar", "organizar", "llevar", "levantar", "empujar", "tirar",
    "balancear", "girar", "atornillar", "desatornillar", "clavar", "alisar", "ajustar",
    "instalar", "subir", "compartir", "viajar", "obtener", "tener", "hacer", "necesitar",
    "querer", "desinstalar", "presionar", "esbozar", "fotografiar", "grabar", "jugar", "ver",
    "descargar", "entender", "recordar", "olvidar", "ayudar", "dar", "mostrar", "abrir",
    "cerrar", "iniciar", "terminar"
]


def generate_random_word_pairs(nouns: list, verbs: list, n: int = 500, seed: int = 42) -> list:
    """Generate n random word pairs from the given noun and verb lists."""
    random.seed(seed)
    pairs = []
    for _ in range(n):
        w1 = random.choice(verbs)
        w2 = random.choice(nouns)
        pairs.append((w1, w2))
    return pairs


def get_headlines_from_hf(language: str, n: int = 500) -> list:
    """
    Fetch headlines from HuggingFace datasets.

    English: cc_news dataset
    Chinese: clue (news titles) or similar
    Spanish: mlsum (Spanish news)
    """
    headlines = []

    if language == "en":
        # Use CC-News for English headlines
        try:
            ds = load_dataset("cc_news", split="train", streaming=True)
            for i, item in enumerate(ds):
                if i >= n:
                    break
                if "title" in item and item["title"]:
                    headlines.append(item["title"])
        except Exception as e:
            print(f"Could not load cc_news: {e}")
            # Fallback: use some sample headlines
            print("Using sample headlines instead...")
            sample_headlines = [
                "Scientists discover New Species in Amazon Rainforest",
                "Stock Market Reaches All-Time High",
                "New Study Reveals Benefits of Coffee",
                "Tech Giant Announces Revolutionary Product",
                "Climate Summit Ends with Historic Agreement",
            ]
            headlines = sample_headlines * (n // len(sample_headlines) + 1)
            headlines = headlines[:n]

    elif language == "zh":
        # Use Chinese news dataset
        try:
            ds = load_dataset("jed351/rthk_news", split="train", streaming=True)
            for i, item in enumerate(ds):
                if i >= n:
                    break
                if "headlines" in item and item["headlines"]:
                    headlines.append(item["headlines"])
        except Exception as e:
            print(f"Could not load Chinese news: {e}")
            sample_headlines = [
                "科学家在深海发现新物种",
                "科技公司发布新产品引发关注",
                "研究表明运动有助于改善睡眠",
                "全球气候大会达成重要协议",
                "新能源汽车销量持续增长",
            ]
            headlines = sample_headlines * (n // len(sample_headlines) + 1)
            headlines = headlines[:n]

    elif language == "es":
        # Use Spanish news dataset 
        try: 
            ds = load_dataset("hacktoberfest-corpus-es/colmbian_spanish_news", split="train", streaming=True)
            for i, item in enumerate(ds):
                if i >= n:
                    break
                if "news_title" in item and item["news_title"]:
                    headlines.append(item["news_title"])
        except Exception as e:
            print(f"Could not load Spanish news: {e}")
            sample_headlines = [
                "Científicos descubren nueva especie en la Amazonía",
                "El mercado de valores alcanza máximos históricos",
                "Nuevo estudio revela beneficios del café",
                "Empresa tecnológica anuncia producto revolucionario",
                "Cumbre climática termina con acuerdo histórico",
            ]
            headlines = sample_headlines * (n // len(sample_headlines) + 1)
            headlines = headlines[:n]

    return headlines


# Prompt constructors matching notebook format
def construct_pair_prompt_en(w1: str, w2: str) -> str:
    return f"Generate a funny joke using these two words: '{w1}', '{w2}'. Only respond with the joke and nothing else."


def construct_headline_prompt_en(headline: str) -> str:
    return f"Generate a funny joke related to this headline: '{headline}'. Only respond with the joke and nothing else."


def construct_pair_prompt_zh(w1: str, w2: str) -> str:
    return f"用这两个词生成一个有趣的笑话：'{w1}'、'{w2}'。只回复笑话，不要回复其他内容。"


def construct_headline_prompt_zh(headline: str) -> str:
    return f"根据这个标题生成一个有趣的笑话：'{headline}'。只回复笑话，不要回复其他内容。"


def construct_pair_prompt_es(w1: str, w2: str) -> str:
    return f"Genera un chiste gracioso usando estas dos palabras: '{w1}', '{w2}'. Solo responde con el chiste y nada más."


def construct_headline_prompt_es(headline: str) -> str:
    return f"Genera un chiste gracioso relacionado con este titular: '{headline}'. Solo responde con el chiste y nada más."


def prepare_data(
    language: str,
    test_tsv_path: str,
    data_dir: str = "../data",
    n_word_pairs: int = 500,
    n_headlines: int = 500,
):
    """
    Prepare GRPO training data for a specific language.

    Args:
        language: 'en', 'zh', or 'es'
        test_tsv_path: Path to TSV file with test word pairs/headlines
        data_dir: Output directory for parquet files
        n_word_pairs: Number of random word pair prompts for training
        n_headlines: Number of headline prompts for training
    """
    # Language-specific settings
    if language == "en":
        nouns = ENGLISH_NOUNS
        verbs = ENGLISH_VERBS
        pair_prompt_fn = construct_pair_prompt_en
        headline_prompt_fn = construct_headline_prompt_en
        suffix = ""
    elif language == "zh":
        nouns = CHINESE_NOUNS
        verbs = CHINESE_VERBS
        pair_prompt_fn = construct_pair_prompt_zh
        headline_prompt_fn = construct_headline_prompt_zh
        suffix = "_zh"
    elif language == "es":
        nouns = SPANISH_NOUNS
        verbs = SPANISH_VERBS
        pair_prompt_fn = construct_pair_prompt_es
        headline_prompt_fn = construct_headline_prompt_es
        suffix = "_es"
    else:
        raise ValueError(f"Unknown language: {language}")

    # Generate training data
    print(f"Generating {n_word_pairs} random word pair prompts for {language}...")
    train_pairs = generate_random_word_pairs(nouns, verbs, n=n_word_pairs)
    pair_prompts = [pair_prompt_fn(w1, w2) for w1, w2 in train_pairs]

    print(f"Fetching {n_headlines} headlines from HuggingFace...")
    headlines = get_headlines_from_hf(language, n=n_headlines)
    headline_prompts = [headline_prompt_fn(h) for h in headlines]

    train_prompts = pair_prompts + headline_prompts
    random.shuffle(train_prompts)
    train_df = pd.DataFrame({"prompt": train_prompts})

    # Load test data from TSV
    print(f"Loading test data from {test_tsv_path}...")
    test_data = pd.read_csv(test_tsv_path, delimiter="\t")

    test_prompts = []
    # Process word pairs and headlines from TSV (matching notebook format)
    for _, row in test_data.iterrows():
        w1 = row.get("word1", "-")
        w2 = row.get("word2", "-")
        headline = row.get("headline", "-")

        if w1 != "-" and w2 != "-":
            test_prompts.append(pair_prompt_fn(w1, w2))
        elif headline != "-":
            test_prompts.append(headline_prompt_fn(headline))

    test_df = pd.DataFrame({"prompt": test_prompts})

    # Save to parquet
    train_path = os.path.join(data_dir, f"rl_df_train{suffix}.parquet")
    test_path = os.path.join(data_dir, f"rl_df_test{suffix}.parquet")

    Dataset.from_pandas(train_df).to_parquet(train_path)
    Dataset.from_pandas(test_df).to_parquet(test_path)

    print(f"Created {len(train_df)} training prompts -> {train_path}")
    print(f"Created {len(test_df)} test prompts -> {test_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare GRPO training data")
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["en", "zh", "es"],
        help="Language to prepare data for",
    )
    parser.add_argument(
        "--test_tsv",
        type=str,
        required=True,
        help="Path to TSV file with test word pairs/headlines",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--n_word_pairs",
        type=int,
        default=500,
        help="Number of random word pair prompts for training",
    )
    parser.add_argument(
        "--n_headlines",
        type=int,
        default=500,
        help="Number of headline prompts for training",
    )
    args = parser.parse_args()

    prepare_data(
        language=args.language,
        test_tsv_path=args.test_tsv,
        data_dir=args.data_dir,
        n_word_pairs=args.n_word_pairs,
        n_headlines=args.n_headlines,
    )

    print("\nDone!")
