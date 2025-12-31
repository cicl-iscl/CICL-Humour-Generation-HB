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

# Common nouns for random word pair generation (English)
ENGLISH_NOUNS = [
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
    "airplane", "helicopter", "submarine", "rocket", "spaceship", "bicycle", "motorcycle", "boat", "train"
]

# Chinese nouns for random word pair generation
CHINESE_NOUNS = [
    "猫", "狗", "房子", "汽车", "树", "手机", "书", "椅子", "桌子", "电脑",
    "咖啡", "披萨", "香蕉", "苹果", "橙子", "水", "火", "冰", "太阳", "月亮",
    "星星", "云", "雨", "雪", "风", "山", "河", "海", "沙滩", "森林",
    "鸟", "鱼", "马", "牛", "猪", "鸡", "鸭", "兔子", "老鼠", "大象",
    "狮子", "老虎", "熊", "猴子", "蛇", "青蛙", "蝴蝶", "蜘蛛", "蚂蚁", "蜜蜂",
    "医生", "老师", "律师", "厨师", "飞行员", "护士", "艺术家", "音乐家", "作家", "工程师",
    "机器人", "外星人", "僵尸", "吸血鬼", "鬼", "魔法师", "忍者", "海盗", "骑士", "龙",
    "锤子", "螺丝刀", "梯子", "桶", "绳子", "钉子", "刷子", "镜子", "钟", "灯",
    "衬衫", "裤子", "鞋子", "帽子", "眼镜", "手表", "戒指", "项链", "雨伞", "包",
    "蛋糕", "饼干", "面包", "奶酪", "黄油", "牛奶", "鸡蛋", "培根", "沙拉", "汤",
    "微波炉", "冰箱", "电视", "键盘", "耳机", "相机", "电池", "充电器",
    "铅笔", "橡皮", "剪刀", "订书机", "笔记本", "信封", "邮票", "日历", "地图",
    "吉他", "钢琴", "鼓", "小提琴", "小号", "长笛", "萨克斯", "麦克风", "扬声器",
    "足球", "篮球", "网球", "高尔夫", "棒球", "曲棍球", "游泳", "滑雪", "冲浪",
    "飞机", "直升机", "潜水艇", "火箭", "宇宙飞船", "自行车", "摩托车", "船", "火车"
]

# Spanish nouns for random word pair generation
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
]


def generate_random_word_pairs(nouns: list, n: int = 500, seed: int = 42) -> list:
    """Generate n random word pairs from the given noun list."""
    random.seed(seed)
    pairs = []
    for _ in range(n):
        w1, w2 = random.sample(nouns, 2)
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
            ds = load_dataset("seamew/CNews", split="train", streaming=True)
            for i, item in enumerate(ds):
                if i >= n:
                    break
                if "title" in item and item["title"]:
                    headlines.append(item["title"])
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
        # Use Spanish news dataset (mlsum)
        try:
            ds = load_dataset("mlsum", "es", split="train", streaming=True)
            for i, item in enumerate(ds):
                if i >= n:
                    break
                if "title" in item and item["title"]:
                    headlines.append(item["title"])
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
    return f"Generate a funny joke using these two words: '{w1}', '{w2}'."


def construct_headline_prompt_en(headline: str) -> str:
    return f"Generate a funny joke related to this headline: '{headline}'."


def construct_pair_prompt_zh(w1: str, w2: str) -> str:
    return f"用这两个词生成一个有趣的笑话：'{w1}'、'{w2}'。"


def construct_headline_prompt_zh(headline: str) -> str:
    return f"根据这个标题生成一个有趣的笑话：'{headline}'。"


def construct_pair_prompt_es(w1: str, w2: str) -> str:
    return f"Genera un chiste gracioso usando estas dos palabras: '{w1}', '{w2}'."


def construct_headline_prompt_es(headline: str) -> str:
    return f"Genera un chiste gracioso relacionado con este titular: '{headline}'."


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
        pair_prompt_fn = construct_pair_prompt_en
        headline_prompt_fn = construct_headline_prompt_en
        suffix = ""
    elif language == "zh":
        nouns = CHINESE_NOUNS
        pair_prompt_fn = construct_pair_prompt_zh
        headline_prompt_fn = construct_headline_prompt_zh
        suffix = "_zh"
    elif language == "es":
        nouns = SPANISH_NOUNS
        pair_prompt_fn = construct_pair_prompt_es
        headline_prompt_fn = construct_headline_prompt_es
        suffix = "_es"
    else:
        raise ValueError(f"Unknown language: {language}")

    # Generate training data
    print(f"Generating {n_word_pairs} random word pair prompts for {language}...")
    train_pairs = generate_random_word_pairs(nouns, n=n_word_pairs)
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
