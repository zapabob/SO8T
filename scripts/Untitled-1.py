# -*- coding: utf-8 -*-
# ファイル名: make_proposal_pptx.py
# 概要: 提案書のPPTXを作成するスクリプト
# 作者: 峯岸 亮
# 日付: 2025/11/09
# バージョン: 1.0
# 使用方法: python make_proposal_pptx.py
# 依存ライブラリ: pptx
# ライセンス: MIT
# 著作権: 2025 峯岸 亮
# 連絡先: 峯岸 亮 <r.minegishi1@gmail.com>
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
import python_pptx as pptx
from python_pptx import Presentation
from python_pptx.util import Inches, Pt
from python_pptx.enum.text import PP_ALIGN
from python_pptx.enum.shapes import MSO_SHAPE
from python_pptx.dml.color import RGBColor



def add_textbox(slide, left, top, width, height, text, font_size=16, bold=False, color=(0,0,0), align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(*color)
    p.alignment = align
    return tb

def add_labeled_box(slide, left, top, width, height, label,
                    fill_rgb=(255,255,255), line_rgb=(34,34,34), font_size=16):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shp.fill.solid()
    shp.fill.fore_color.rgb = RGBColor(*fill_rgb)
    shp.line.color.rgb = RGBColor(*line_rgb)
    tf = shp.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = label
    run.font.size = Pt(font_size)
    p.alignment = PP_ALIGN.CENTER
    return shp

def add_arrow(slide, x1, y1, x2, y2, color=(34,34,34), width_pt=2):
    conn = slide.shapes.add_connector(1, x1, y1, x2, y2)  # straight connector
    conn.line.color.rgb = RGBColor(*color)
    conn.line.width = Pt(width_pt)
    conn.line.end_arrowhead = True
    return conn

def build_presentation():
    prs = Presentation()
    prs.slide_width  = Inches(13.33)  # 16:9
    prs.slide_height = Inches(7.5)

    # 1) Title
    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = "クローズド LLMOps／AIエージェント基盤 提案書"
    s.placeholders[1].text = (
        "四値判定（ALLOW / ESCALATION / DENY / REFUSE）\n"
        "学習時作用→推論は標準重み（焼き込み）\n"
        "提出者：峯岸 亮（放送大学 在籍／研究開発・実装担当）"
    )

    # 2) Executive Summary
    s = prs.slides.add_slide(prs.slide_layouts[5])
    add_textbox(s, Inches(0.7), Inches(0.5), Inches(12), Inches(0.6),
                "エグゼクティブサマリー", font_size=28, bold=True)
    summary = (
        "日本語特化の基盤モデルを完全クローズド環境で運用し、四値判定を備えた安全指向AIエージェントへ発展させる。"
        " 技術中核は『学習時のみ作用し、推論時は標準重みに等価変換（焼き込み）して実行』する学習強化機構である。"
        " 配備時は標準推論グラフのみで稼働し、監査容易性・長期保守性・標準ランタイム互換を成立。"
        " 全自動パイプラインは分岐／スキップ制御・署名検証・AB並走を備え、誤許可率の最小化と実運用の再現性を両立する。"
    )
    add_textbox(s, Inches(0.7), Inches(1.3), Inches(12), Inches(3.5), summary, font_size=18)

    # 3) Architecture (abstract, no formula)
    s = prs.slides.add_slide(prs.slide_layouts[5])
    add_textbox(s, Inches(0.7), Inches(0.5), Inches(12), Inches(0.6),
                "アーキテクチャ（抽象化・数式なし）", font_size=28, bold=True)
    add_labeled_box(s, Inches(0.6), Inches(1.2), Inches(12.1), Inches(1.0),
                    "データ境界層（閉域）：機密データ／業務ログ／ルール文書／FAQ／法令集（分類・マスキング済）",
                    fill_rgb=(255,255,255), line_rgb=(34,34,34), font_size=16)
    add_labeled_box(s, Inches(0.6), Inches(2.4), Inches(12.1), Inches(0.9),
                    "知識層（RAG + CAG）：閉域ベクタDB＋キャッシュ拡張生成、SRS（忘却曲線）で鮮度管理",
                    fill_rgb=(240,244,255), line_rgb=(74,103,255), font_size=16)
    add_labeled_box(s, Inches(0.6), Inches(3.5), Inches(12.1), Inches(1.2),
                    "学習・適応層：省メモリ学習（LoRA/QLoRA）＋学習時作用モジュール（NDA 対象）→焼き込み→標準重み",
                    fill_rgb=(255,247,236), line_rgb=(251,106,0), font_size=16)
    add_labeled_box(s, Inches(0.6), Inches(4.9), Inches(12.1), Inches(1.0),
                    "推論層（ローカル）：量子化済み標準モデル（署名付き）＋四値判定（較正済）",
                    fill_rgb=(247,255,240), line_rgb=(85,166,48), font_size=16)
    add_labeled_box(s, Inches(0.6), Inches(6.0), Inches(12.1), Inches(1.0),
                    "運用・監査層：分岐／スキップ／署名検証／AB並走／監査ログ／SLO・KPI ダッシュボード",
                    fill_rgb=(254,249,195), line_rgb=(212,166,0), font_size=16)
    add_arrow(s, Inches(6.65), Inches(2.2), Inches(6.65), Inches(2.4))
    add_arrow(s, Inches(6.65), Inches(3.3), Inches(6.65), Inches(3.5))
    add_arrow(s, Inches(6.65), Inches(4.7), Inches(6.65), Inches(4.9))
    add_arrow(s, Inches(6.65), Inches(5.9), Inches(6.65), Inches(6.0))
    add_textbox(s, Inches(8.8), Inches(3.9), Inches(3.6), Inches(0.6),
                "※ 詳細原理は NDA 後の共同研究対象", font_size=14, color=(70,70,70))
    add_textbox(s, Inches(8.8), Inches(5.0), Inches(3.6), Inches(0.6),
                "※ 推論期は標準重みのみ（カスタムOp不要）", font_size=14, color=(70,70,70))

    # 4) SOP
    s = prs.slides.add_slide(prs.slide_layouts[5])
    add_textbox(s, Inches(0.7), Inches(0.5), Inches(12), Inches(0.6),
                "モデル更新 SOP（標準運用手順・抄）", font_size=28, bold=True)
    sop = (
        "1) 入力データ・ルール・ポリシーの差分検査。閾値未満なら再学習をスキップし較正のみ更新。\n"
        "2) 閾値超の差分に対して事後学習を実行。学習時作用モジュールを学習期のみ有効化し、収束後に焼き込みで標準重みへ等価変換。\n"
        "3) 量子化・電子署名を付与し、配備前後で署名検証。不一致時は自動停止。\n"
        "4) 検証ゲートで Macro-F1・誤許可率・ECE・長文劣化・分布外指標を確認。KPI 達成時のみ配備。\n"
        "5) 配備後は較正セットで温度・閾値を最適化。AB 並走の監視で問題あれば即時ロールバック。\n"
        "6) 変更内容は監査ログへ記録。定例レビューで SLO/KPI・署名検証率・再学習スキップ率を報告。"
    )
    add_textbox(s, Inches(0.7), Inches(1.3), Inches(12), Inches(5.3), sop, font_size=18)

    # 5) KPI dashboard
    s = prs.slides.add_slide(prs.slide_layouts[5])
    add_textbox(s, Inches(0.7), Inches(0.5), Inches(12), Inches(0.6),
                "KPI ダッシュボード（抄）", font_size=28, bold=True)
    add_labeled_box(s, Inches(0.6),  Inches(1.2), Inches(3.3), Inches(1.7),
                    "誤許可率（日/7日/30日）", fill_rgb=(238,247,255), line_rgb=(74,103,255))
    add_labeled_box(s, Inches(4.1),  Inches(1.2), Inches(3.3), Inches(1.7),
                    "Macro-F1（四値）", fill_rgb=(238,247,255), line_rgb=(74,103,255))
    add_labeled_box(s, Inches(7.6),  Inches(1.2), Inches(3.3), Inches(1.7),
                    "ECE / 温度較正", fill_rgb=(245,255,240), line_rgb=(85,166,48))
    add_labeled_box(s, Inches(11.1), Inches(1.2), Inches(3.1), Inches(1.7),
                    "分布外判定（ESC/REF）", fill_rgb=(255,247,236), line_rgb=(251,106,0))
    add_labeled_box(s, Inches(0.6),  Inches(3.2), Inches(6.0), Inches(2.2),
                    "混同行列（四値）", fill_rgb=(255,255,255), line_rgb=(34,34,34))
    add_labeled_box(s, Inches(6.9),  Inches(3.2), Inches(7.3), Inches(2.2),
                    "AB並走：現行 vs 新版（差分）", fill_rgb=(255,255,255), line_rgb=(34,34,34))
    add_labeled_box(s, Inches(0.6),  Inches(5.6), Inches(4.1), Inches(1.6),
                    "署名検証 合格率", fill_rgb=(255,255,255), line_rgb=(34,34,34))
    add_labeled_box(s, Inches(5.0),  Inches(5.6), Inches(4.1), Inches(1.6),
                    "再学習スキップ率", fill_rgb=(255,255,255), line_rgb=(34,34,34))
    add_labeled_box(s, Inches(9.4),  Inches(5.6), Inches(4.8), Inches(1.6),
                    "ロールバック発生 / MTTR", fill_rgb=(255,255,255), line_rgb=(34,34,34))

    # 6) Security & Governance
    s = prs.slides.add_slide(prs.slide_layouts[5])
    add_textbox(s, Inches(0.7), Inches(0.5), Inches(12), Inches(0.6),
                "セキュリティ／ガバナンス（抄）", font_size=28, bold=True)
    sec = (
        "・完全クローズド運用：外部送信なし。出力はポリシー検査＋署名検証を通過したもののみ。\n"
        "・監査ログ：入力・出力・設定変更・モデル切替・署名検証を不可逆に記録。\n"
        "・データ境界：RAG も含め閉域データのみを対象。DLP・認可・鍵管理を一体化。\n"
        "・プロンプト注入対策：システムプロンプト固定、危険語彙正規化、出力の生食禁止、ESC/REF 優先。"
    )
    add_textbox(s, Inches(0.7), Inches(1.3), Inches(12), Inches(5.3), sec, font_size=18)

    # 7) Schedule (Gantt-like)
    s = prs.slides.add_slide(prs.slide_layouts[5])
    add_textbox(s, Inches(0.7), Inches(0.5), Inches(12), Inches(0.6),
                "スケジュール（概略・8週間）", font_size=28, bold=True)
    add_textbox(s, Inches(0.7), Inches(1.2), Inches(12), Inches(0.4),
                "W1-2 要件/監査/環境｜W3-4 PoC→焼き込み→配備→較正｜W5-6 分岐/スキップ/AB定着｜W7-8 拡張と制度化", font_size=16)

    y = 2.0
    def bar(x, w, color):
        r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(0.5))
        r.fill.solid(); r.fill.fore_color.rgb = RGBColor(*color)
        r.line.color.rgb = RGBColor(60,60,60)

    bar(0.8, 3.0, (180,220,255)); add_textbox(s, Inches(0.85), Inches(y+0.6), Inches(3.0), Inches(0.3), "要件確定・監査設計・環境整備", font_size=14)
    bar(3.2, 3.2, (200,255,200)); add_textbox(s, Inches(3.25), Inches(y+0.6), Inches(3.2), Inches(0.3), "PoC 学習→焼き込み→量子化→署名→配備→較正", font_size=14)
    bar(6.6, 3.0, (255,220,180)); add_textbox(s, Inches(6.65), Inches(y+0.6), Inches(3.0), Inches(0.3), "分岐/スキップ導入・AB並走・Runbook 固定", font_size=14)
    bar(9.8, 2.8, (255,235,150)); add_textbox(s, Inches(9.85), Inches(y+0.6), Inches(2.8), Inches(0.3), "業務拡張・KPI再設計・監査レビュー制度化", font_size=14)

    # 8) Risks & Mitigations
    s = prs.slides.add_slide(prs.slide_layouts[5])
    add_textbox(s, Inches(0.7), Inches(0.5), Inches(12), Inches(0.6),
                "リスクと対策（抄）", font_size=28, bold=True)
    risk = (
        "・分布外や方針逸脱：エスカレーション優先と四値しきい値最適化で回避。\n"
        "・量子化劣化：温度スケーリングと再較正で ECE を抑制。AB 並走で影響最小化。\n"
        "・データ陳腐化：SRS によるキャッシュ鮮度管理で再学習の乱発を防止。\n"
        "・保守性：焼き込みにより標準ランタイム互換を維持、長期運用でベンダロックを緩和。"
    )
    add_textbox(s, Inches(0.7), Inches(1.3), Inches(12), Inches(5.3), risk, font_size=18)

    # 9) Next Actions
    s = prs.slides.add_slide(prs.slide_layouts[5])
    add_textbox(s, Inches(0.7), Inches(0.5), Inches(12), Inches(0.6),
                "次アクション（NDA 前提）", font_size=28, bold=True)
    nexts = (
        "1) NDA 締結（学習時作用モジュールの詳細・共同研究範囲を確定）。\n"
        "2) データ・評価指標（誤許可率・Macro-F1・ECE 等）の数値合意。\n"
        "3) PoC 閉域環境（GPU/ストレージ/ログ）とスケジュールを確定。\n"
        "4) 知財・学術の取り扱い（共同出願・共同発表）を合意。"
    )
    add_textbox(s, Inches(0.7), Inches(1.3), Inches(12), Inches(5.3), nexts, font_size=18)

    return prs

if __name__ == "__main__":
    prs = build_presentation()
    out = "公共向け提案書_四値判定LLMOps_抽象化設計_v1.pptx"
    prs.save(out)
    print(f"saved: {out}")
