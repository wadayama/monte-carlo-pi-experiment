# 科学技術計算のためのPython開発ガイドライン

## 1. はじめに

### 1.1. 本ガイドラインの目的
この文書は、本プロジェクトで実施する科学技術計算の品質と効率を高めるための開発ガイドラインです。コードの**正確性 (Correctness)**、**可読性 (Readability)**、そして**再現性 (Reproducibility)** の三つの柱を最も重視します。

### 1.2. 基本原則
* **正確さ > 速さ**: 科学技術計算では、計算速度よりも結果の正確性が優先されます。
* **未来の自分や他人のために**: コードは一度書くだけでなく、後から何度も読まれることを意識します。
* **いつでも、誰でも再現可能に**: 10年後でも同じ環境で同じ実験をすぐに再現できる状態を目指します。

---

## 2. 開発環境の構築と管理

**一貫性のある開発環境は、再現性の第一歩です。`uv` を用いて、プロジェクトごとに独立した環境を構築・管理します。**

### 2.1. 仮想環境
プロジェクトのルートディレクトリで以下のコマンドを実行し、仮想環境を作成します。
```bash
# 仮想環境の作成
uv venv

# 仮想環境のアクティベート (Windows/Linux/macOS共通)
source .venv/bin/activate
```

### 2.2. パッケージ管理
パッケージ管理は `uv` に統一し、`pip` や `conda` との併用は避けてください。依存関係は `pyproject.toml` で一元管理します。再現性を担保するため、pyproject.toml にてプロジェクトが要求するPythonのバージョンを明記します。

* **パッケージの追加**:
    ```bash
    # 実行に必要なパッケージを追加
    uv add numpy pandas
    
    # 開発時にのみ必要なパッケージ (テスト、フォーマッタ等) を追加
    uv add -d pytest ruff-lsp
    ```
* **環境の同期**:
    `pyproject.toml` と `uv.lock` ファイルがあれば、以下のコマンドで誰でも同じ環境を再現できます。
    ```bash
    uv sync --frozen
    ```
* **ツールの実行**:
    インストールしたツールは `uv run` を使って実行します。
    ```bash
    uv run pytest
    uv run ruff format .
    ```

---

## 3. コードの品質を保つ規約

### 3.1. 静的解析による品質保証
**人間のレビューの前に、ツールによる自動チェックを最大限活用します。**

* **フォーマット**: `ruff format` を使用して、コードスタイルを統一します。
* **リンティング**: `ruff check` で潜在的なバグや非推奨な書き方を検出します。
* **型チェック**:
    * コードには可能な限り**型ヒント (Type Hints)** を付与します。
    * `pyright` を用いて、型ヒントの整合性を静的に検証します。
    
**※ これらツールの設定は `pyproject.toml` に集約し、チームで共有することを推奨します。**

### 3.2. テストによる正確性の担保
**「重要な関数」だけでなく、ロジックを含む全ての関数にテストを書くことを原則とします。**

* **テストフレームワーク**: `pytest` を利用します。
* **テストの原則**:
    * **小さく、集中**: 一つのテスト関数では、一つのことだけを検証します。
    * **境界値**: 正常系だけでなく、異常系や境界値のテストケースを用意します。
    * **副作用の分離**: 乱数生成、現在時刻の取得、ファイル入出力といった「副作用」を含む処理は、ロジックの中心部から分離して設計します。これにより、ロジック部分のテストが容易になります。



### 3.3. 可読性の向上
* **ドキュメンテーション文字列 (Docstring)**:
    * 全ての公開モジュール、関数、クラス、メソッドにはDocstringを記載します。
    * **スタイルはNumpyスタイルを推奨します。** 引数、返り値、処理内容が明確になります。
    * **Docstringは英語で記述します。** 国際的な共同作業や将来的な利用を考慮し、英語での記述を標準とします。
* **コメント**: 「なぜ」そのコードが必要なのか、複雑なロジックの意図を説明するためにコメントを活用します。コードを見れば分かる「何」を説明するコメントは不要です。
    * **コメントも英語で記述します。** チーム開発や国際的な環境での利用を前提とします。
* **命名規則**: **PEP 8 に準拠した、分かりやすい変数名・関数名を心がけてください。**
    * **変数名・関数名は英語を使用します。** ローマ字での命名は避け、適切な英単語を使用してください。

### 3.4. 国際化とユーザビリティ
* **プログラム出力の言語**:
    * **コンソール出力、ログメッセージ、エラーメッセージは英語で記述します。**
    * プログラムの実行結果や進捗表示も英語を標準とします。
    * 命名: 変数名、関数名、クラス名（※ ローマ字表記は不可）
* **グラフとプロット**:
    * **軸ラベル、タイトル、凡例は英語で記述します。**
    * matplotlib等で作成する図表も国際的な利用を考慮し、英語表記を標準とします。
* **ファイル出力**:
    * CSVのヘッダー、PDFのタイトル等も英語で統一します。

### 3.5. 関数設計の原則

### 3.5.1. 小さな関数への分割
保守性と可読性の観点から、ひとつの関数はなるべく小さく保ちます。
* Single Responsibility Principle: 一つの関数は一つの責任のみを持つ
* 関数の長さ: 一般的に20-30行以内を目安とし、画面に収まる範囲で設計
* 複雑な処理の分解: 複雑なアルゴリズムは意味のある単位で関数に分割

#### Bad: 長大で複雑な関数
```
def analyze_data(data_path: str) -> dict:
    """Analyze data from file (too complex)."""
    # データ読み込み (10行)
    # データ前処理 (15行)
    # 統計計算 (20行)
    # 可視化 (15行)
    # 結果保存 (10行)
    pass  # 計70行の複雑な処理
```
#### Good: 小さな関数に分割
```
def load_data(data_path: str) -> pd.DataFrame:
    """Load data from specified path."""
    return pd.read_csv(data_path)

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the input data."""
    # 前処理ロジック
    return cleaned_data

def calculate_statistics(data: pd.DataFrame) -> dict:
    """Calculate basic statistical measures."""
    # 統計計算ロジック
    return statistics

def create_visualization(data: pd.DataFrame, stats: dict) -> plt.Figure:
    """Create visualization plots."""
    # 可視化ロジック
    return figure

def save_results(stats: dict, figure: plt.Figure, output_dir: Path) -> None:
    """Save analysis results to files."""
    # 保存ロジック
    pass

def analyze_data(data_path: str, output_dir: Path) -> dict:
    """Analyze data with clear workflow."""
    data = load_data(data_path)
    cleaned_data = preprocess_data(data)
    stats = calculate_statistics(cleaned_data)
    figure = create_visualization(cleaned_data, stats)
    save_results(stats, figure, output_dir)
    return stats

```
### 3.5.2 参照透過性と副作用の分離
計算ロジック（純粋関数）とI/O処理（副作用を持つ関数）を明確に分離します。
#### 参照透過性を持つ純粋関数の特徴
* 同じ入力に対して常に同じ出力: 関数の実行結果が入力のみに依存
* 副作用なし: グローバル変数の変更、ファイル操作、ネットワーク通信などを行わない
* テストしやすい: 入力と出力の関係が明確で、モックが不要

### 3.5.3. 関数設計のベストプラクティス

* 関数名は動詞で始める: calculate_, process_, validate_ など
* 引数の順序: 重要な引数を先に、オプション引数を後に
* デフォルト引数: 可変オブジェクト（リスト、辞書）をデフォルト値にしない
* 型ヒントの活用: 入力・出力の型を明示して、関数の契約を明確にする

### 3.5.4 グローバル変数の使用を極力避ける

参照透過な部分を多くするため、グローバル変数の利用は極力控えます。
グローバル変数は関数の動作を予測困難にし、テストを困難にし、並列処理時の競合状態を引き起こす可能性があります。
ただし，物理定数やプログラム内で不変な定数などのグローバル変数は，コードの可読性を向上させるために使用しても良いとします．

## 4. 再現性の確保

### 4.1. バージョン管理
* **全てのコードと設定ファイルは Git でバージョン管理します。**
* 最初にgh コマンドで，Githubにプライベートレポジトリを作成してください．
* 変更は意味のある単位で、`feat:`, `fix:`, `docs:`, `test:` のようなプレフィックスを付けた分かりやすいメッセージと共にコミットしてください。
* コミット前にローカルでチェック: uv run ruff format ., uv run ruff check ., uv run pyright, uv run pytest を自動実行して問題点があれば修正してください。
* また「テストしてください」という指示があった場合には，テストを含めた以下の実行をおこなってください：uv run ruff format ., uv run ruff check ., uv run pyright, uv run pytest

### 4.2. 実験パラメータの管理
* 学習率、データパス、繰り返し回数などの主要な実験パラメータは、コードに直接書き込まず（ハードコーディングしない）、**YAMLファイルに分離して管理します。**
* **乱数を用いる場合は、そのシード値もYAMLファイルに記載し**、コードで読み込んで設定することで、結果の再現性を保証します。

### 4.3. データ管理
* **入力データと出力データは、明確なディレクトリ構造で管理します。** 下記はその一例です．
    ```
    project-root/
    ├── pyproject.toml          # プロジェクト設定・依存関係
    ├── uv.lock                 # ロックファイル
    ├── README.md               # プロジェクト説明
    ├── .gitignore              # Git除外設定
    ├── config/                 # 実験パラメータ設定
    │   ├── default.yaml
    │   └── experiment_*.yaml
    ├── data/
    │   ├── raw/                # 元データ（読み取り専用）
    │   ├── interim/            # 中間処理データ
    │   └── processed/          # 最終的な入力データ
    ├── src/
    │   ├── __init__.py
    │   ├── data/               # データ処理モジュール
    │   ├── models/             # モデル定義
    │   ├── experiments/        # 実験スクリプト
    │   └── utils/              # ユーティリティ関数
    ├── tests/                  # テストコード
    │   ├── __init__.py
    │   ├── test_data/
    │   ├── test_models/
    │   └── test_utils/
    ├── outputs/                # 実験結果出力
    │   ├── 2025-06-12_experiment-A/
    │   │   ├── config.yaml     # 使用した設定のコピー
    │   │   ├── results.csv
    │   │   ├── figures/
    │   │   └── logs/
    │   └── ...
    └── notebooks/              # 探索的データ解析用
        ├── 01_data_exploration.ipynb
        └── 02_result_analysis.ipynb
    ```

* 実験結果の出力ファイルや、出力されるログファイル内に、
実験を実行した時点でのGitのコミットハッシュ を含めることを推奨します。
これにより、どのコードバージョンがその結果を生成したかを一意に特定できます。
コミットハッシュがあれば，git checkout コマンドでそのコミット時のコードに戻ることができます。
以下は参考コードになります．
```Python
import subprocess
import logging

def get_git_commit_hash() -> str:
    """
    現在のGitリポジトリのコミットハッシュを取得する。
    未コミットの変更がある場合は、ハッシュの末尾に '-dirty' を付与する。
    """
    try:
        # git rev-parse HEAD コマンドで最新のコミットハッシュを取得
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).strip().decode('utf-8')

        # git status --porcelain コマンドで未コミットの変更がないかチェック
        # 何か変更があれば出力があり、クリーンな状態なら出力はない
        status = subprocess.check_output(
            ['git', 'status', '--porcelain']
        ).strip().decode('utf-8')

        # 未コミットの変更があれば、再現性が不完全であることを示す '-dirty' を付ける
        if status:
            return f"{commit_hash}-dirty"
        else:
            return commit_hash

    except (subprocess.CalledProcessError, FileNotFoundError):
        # Gitリポジトリでない、またはgitコマンドが見つからない場合
        return "not-a-git-repo"

# --- 使い方 ---

# 1. 実験の開始時にコミットハッシュを取得
commit_id = get_git_commit_hash()

# 2. ログに出力する
logging.info(f"Experiment running on git commit: {commit_id}")
if commit_id.endswith('-dirty'):
    logging.warning("Working directory has uncommitted changes. Reproducibility might be compromised.")


```


### 4.4. 実験環境の再現
* uv sync --frozen を実行することで，実験環境を再現できるようにすること．
* このコマンドは，uv.lock ファイルを参照して，必要なパッケージをインストールする．

---

## 5. ドキュメンテーション

### 5.1. README.md の作成
**プロジェクトには必ず `README.md` を作成し、第三者が容易に理解・実行できるよう情報を整理します。**

#### 5.1.1. README.md に含めるべき内容
1. **プロジェクトの概要**
   * 何を目的とした実験・プログラムなのかを簡潔に説明
   * 背景や理論的根拠があれば記載

2. **ファイル構成**
   * プロジェクト内の主要ファイルの役割を説明
   * ディレクトリ構造を視覚的に示す

3. **環境構築手順**
   * 前提条件（Pythonバージョン、必要ツール）
   * `uv sync` を使った環境セットアップ手順
   * 具体的なコマンド例を記載

4. **実行方法**
   * メインプログラムの実行コマンド
   * 実行例と期待される出力の例示
   * パラメータの設定方法

5. **テスト実行方法**
   * `pytest` を使ったテストの実行コマンド
   * テストの内容や目的の説明

6. **コード品質チェック**
   * `ruff format`, `ruff check`, `pyright` の実行方法
   * 開発時の品質保証手順

#### 5.1.2. README.md 作成時の注意点
* **UTF-8エンコードで保存**: 日本語を含む場合は必ずUTF-8エンコードで保存する
* **マークダウン記法の活用**: コードブロック、リスト、見出しを適切に使用
* **具体例の提示**: 抽象的な説明ではなく、実際のコマンドや出力例を示す
* **更新の継続**: プロジェクトの変更に合わせてREADMEも更新する

#### 5.1.3. 推奨構成
```markdown
# プロジェクト名

## 概要
## 実験内容（該当する場合）
## ファイル構成
## 環境構築
### 前提条件
### セットアップ手順
## 実行方法
### 基本実行
### 実行例
## パラメータ設定（該当する場合）
## テスト実行
## コード品質チェック
## プログラム構造（複雑な場合）
## 理論的背景（科学技術計算の場合）
## 注意点
## ライセンス
```

### 5.2. エンコーディングの注意
* **必ずUTF-8エンコードを使用**: 特に日本語を含むファイルでは、エディタの設定を確認
* **BOMなしUTF-8を推奨**: 一部のツールでBOM付きファイルが問題となる場合がある
* **作成後の確認**: `file` コマンドでエンコーディングを確認することを推奨
  ```bash
  file README.md  # "Unicode text, UTF-8 text" と表示されるべき
  ```
---
