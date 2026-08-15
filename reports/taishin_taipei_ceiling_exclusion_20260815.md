# 台新－台北 D1 漲停開盤改為截尾的重建紀錄

日期：2026-08-15

## 變更

原本 21 筆 D1 開盤剛好等於依 D0 收盤計算之跳動單位漲停價的事件，會往後尋找第一個未封漲停的開盤日，並把該日當成進場日與結算日。本次改為不假設後續有可成交的放空：這些事件標記為 `censored=True`，保留原始 D1 OHLC 與以原始 D1 OHLC 算出的 raw 欄位，但完全不計入已結算／勝率／報酬／KPI 母體。

這與 City-GA／富邦建檔器的 `censored` 處理原則一致；兩者原本以整日平盤 OHLC 標記無法執行的 D1，台新－台北另將「D1 開盤已在漲停價」納入同一個無可驗證開盤成交的 `censored` 欄位。

## 重建稽核

- D0 母體不變：254 筆，日期 2026-04-07 至 2026-08-10。
- D1 開盤漲停而截尾：21/254（8.27%）。
- `censored` 總數：21/254；因此已結算母體為 233 筆。
- 所有 254 筆均為 `exit_kind=d1_close`，且 `exit_date=d1`、`exit_price=d1_close`。
- D1 硬性可交易性稽核：254 → 238（排除 D1 處置）→ 233（再排除 D1 停止先賣後買）；已結算母體為 233 → 213。

## 延後進場與直接排除的比較

「已結算母體」表的期間報酬使用頁面同一日期多筆等權、逐日複利的定義。舊值依舊版實際延後日分組；新值依 D1 分組。

| 口徑 | 已結算 n | 勝率 | 平均單筆放空報酬 | 期間報酬 | Sharpe | 最大回撤 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 舊：漲停開盤後延到第一個未封漲停日 | 254 | 54.33% | +0.566% | +25.25% | 1.72 | -19.12% |
| 新：漲停開盤直接 `censored`／排除 | 233 | 55.79% | +0.821% | +38.71% | 2.40 | -17.77% |

頁面預設還硬性排除 D1 處置與 D1 停止先賣後買；該完整頁面口徑由舊版的 n=233、勝率 57.08%、平均 +0.807%、期間 +37.03%，改為 n=213、勝率 59.15%、平均 +1.124%、期間 +53.71%。這組差異同時反映本次的截尾改正，以及硬性稽核從延後日改回真實 D1 的必要對齊。

以上是歷史描述，未包含交易成本、滑價、實際借券額度或成交量驗證，不能作為可執行性或獲利保證。

## Schema 與頁面對齊

台新－台北已移除所有延後進場欄位：`entry_day`、`entry_open`、`entry_high`、`entry_low`、`entry_close`、`entry_frozen`、`entry_disposal`、`entry_disposal_type`、`entry_is_ceiling`、`day_trade_short_suspended_entry`。建檔後的核心結算欄位與富邦的 plain-D1 schema 相同；唯一額外欄位為 `day_trade_short_suspended_d1`，這是本頁既有且必要的 D1 借券／當沖合法性硬性稽核。離線分點 context 欄位則與富邦現有 CSV 相同。

和 City-GA CSV 的既有差異均非本次延後進場機制：City-GA 用它專屬的 `cz_influence_pct`（台新／富邦用 `influence_pct`），並保留 City-GA 專屬的 `mktcap_pct_rank`；台新則多上述 D1 當沖放空限制欄位。其餘 plain-D1 結算欄位與 context 欄位採相同慣例。

頁面以 `d1_disposal` 與 `day_trade_short_suspended_d1` 做硬性排除，投組日分組改為 `d1`，D1 開盤跳空控制直接使用既有 `gap_pct`。

## 驗證

- `E:\stock\venv\Scripts\python.exe -m py_compile scripts\build_taishin_taipei_events.py pages\17_台新台北獨立分點研究.py`
- 延後進場欄位名稱在 Page 17 的 grep 結果為空。
- CSV schema 比對：相對 `data/fubon_branch_events.csv`，只有上述刻意保留的 `day_trade_short_suspended_d1` 額外欄位；富邦沒有台新缺少的欄位。相對 City-GA 的差異如上一段所列。
