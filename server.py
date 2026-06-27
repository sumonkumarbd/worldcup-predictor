"""
server.py — backend for the World Cup Predictor dashboard.

Run:
    pip install flask --break-system-packages   # plus predictor_core's deps
    python3 server.py
Then open http://localhost:5000

Behaviour:
- Loads + trains once at startup (~5-10s).
- Auto-refreshes (re-downloads data, retrains) every AUTO_REFRESH_HOURS in a
  background thread, so the dashboard stays current without you doing anything.
- "Refresh now" in the UI triggers the same refresh on demand.
"""
import threading
import time
from datetime import datetime, timezone

from flask import Flask, jsonify, request, send_from_directory

try:
    import predictor_core as pc
except KeyboardInterrupt:
    print("\nShutting down — nothing left running.")
    raise SystemExit(0)

AUTO_REFRESH_HOURS = 3

app = Flask(__name__, static_folder="static", static_url_path="")

_lock = threading.Lock()
_state = {"data": None, "last_refreshed": None, "is_refreshing": False, "error": None}


def _do_refresh(force_download):
    with _lock:
        if _state["is_refreshing"]:
            return
        _state["is_refreshing"] = True
        _state["error"] = None
    try:
        result = pc.run_pipeline(force_refresh=force_download)
        with _lock:
            _state["data"] = result
            _state["last_refreshed"] = datetime.now(timezone.utc).isoformat()
    except Exception as e:
        with _lock:
            _state["error"] = str(e)
    finally:
        with _lock:
            _state["is_refreshing"] = False


def _background_loop():
    while True:
        time.sleep(AUTO_REFRESH_HOURS * 3600)
        _do_refresh(force_download=True)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/status")
def status():
    with _lock:
        d = _state["data"]
        return jsonify(
            ready=d is not None,
            last_refreshed=_state["last_refreshed"],
            is_refreshing=_state["is_refreshing"],
            error=_state["error"],
            n_matches=d["n_matches"] if d else None,
            auto_refresh_hours=AUTO_REFRESH_HOURS,
        )


@app.route("/api/refresh", methods=["POST"])
def refresh():
    force_download = request.args.get("download", "true") != "false"
    with _lock:
        if _state["is_refreshing"]:
            return jsonify(started=False, message="already refreshing")
    threading.Thread(target=_do_refresh, args=(force_download,), daemon=True).start()
    return jsonify(started=True)


@app.route("/api/backtest")
def backtest():
    with _lock:
        d = _state["data"]
    if not d:
        return jsonify(error="not ready"), 503
    return jsonify(d["backtest"])


@app.route("/api/predictions")
def predictions():
    with _lock:
        d = _state["data"]
    if not d:
        return jsonify(error="not ready"), 503
    preds = []
    for item in d["predictions"]:
        pred = dict(item)
        explanation = pc.explain_matchup(
            d["gbm"], d["elo_final"], d["form_final"],
            pred["home"], pred["away"],
            neutral=pred.get("neutral", True),
            k=int(pred.get("k", 60)),
            rho=d.get("rho", 0.0)
        )
        pred["home_explanation"] = explanation["home_explanation"]
        pred["away_explanation"] = explanation["away_explanation"]
        pred.pop("neutral", None)
        pred.pop("k", None)
        # is_knockout / p_home_advances / p_away_advances pass through if present
        preds.append(pred)
    return jsonify(preds)


@app.route("/api/teams")
def teams():
    with _lock:
        d = _state["data"]
    if not d:
        return jsonify(error="not ready"), 503
    return jsonify(d["teams"])


@app.route("/api/matchup")
def matchup():
    with _lock:
        d = _state["data"]
    if not d:
        return jsonify(error="not ready"), 503
    home, away = request.args.get("home"), request.args.get("away")
    neutral = request.args.get("neutral", "true") == "true"
    k = int(request.args.get("k", 60))
    is_knockout = request.args.get("is_knockout", "false") == "true"
    if not home or not away:
        return jsonify(error="home and away are required"), 400
    if home == away:
        return jsonify(error="pick two different teams"), 400
    result = pc.predict_matchup(d["gbm"], d["elo_final"], d["form_final"], home, away, neutral=neutral, k=k, rho=d.get("rho", 0.0), is_knockout=is_knockout)
    explanation = pc.explain_matchup(d["gbm"], d["elo_final"], d["form_final"], home, away, neutral=neutral, k=k, rho=d.get("rho", 0.0))
    result["home_explanation"] = explanation["home_explanation"]
    result["away_explanation"] = explanation["away_explanation"]
    return jsonify(result)


if __name__ == "__main__":
    import sys
    try:
        print("Initial load + train (one-off, ~5-10s)...")
        _do_refresh(force_download=True)
        if _state["error"]:
            print("Startup error:", _state["error"])
        threading.Thread(target=_background_loop, daemon=True).start()
    except KeyboardInterrupt:
        print("\nShutting down — nothing left running.")
        sys.exit(0)
    app.run(host="0.0.0.0", port=5000, debug=False)
    print("\nShutting down — nothing left running.")
