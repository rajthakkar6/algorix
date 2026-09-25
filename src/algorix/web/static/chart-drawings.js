/*
 * Chart drawing tools for the stock detail page's candlestick chart.
 *
 * TrendLine / TrendLinePaneView / TrendLinePaneRenderer are a JS port of
 * TradingView's own official reference implementation for lightweight-charts
 * v5 series primitives:
 *   https://github.com/tradingview/lightweight-charts/blob/master/plugin-examples/src/plugins/trend-line/trend-line.ts
 * Ported (TypeScript types dropped, otherwise structurally the same:
 * coordinate conversion via series.priceToCoordinate/timeScale.timeToCoordinate,
 * canvas draw scaled by horizontalPixelRatio/verticalPixelRatio) rather than
 * written from memory -- see PROJECT_SCOPE.md's chart UI section for why
 * that discipline matters for this library.
 *
 * AlgorixDrawingTools at the bottom is the extension point: a future drawing
 * tool (horizontal line, rectangle, fib retracement) is a new registry entry
 * here, never a change to storage or the web routes.
 */

class TrendLinePaneRenderer {
  constructor(p1, p2, text1, text2, options) {
    this._p1 = p1;
    this._p2 = p2;
    this._text1 = text1;
    this._text2 = text2;
    this._options = options;
  }

  draw(target) {
    target.useBitmapCoordinateSpace((scope) => {
      if (this._p1.x === null || this._p1.y === null ||
          this._p2.x === null || this._p2.y === null) return;
      var ctx = scope.context;
      var x1 = Math.round(this._p1.x * scope.horizontalPixelRatio);
      var y1 = Math.round(this._p1.y * scope.verticalPixelRatio);
      var x2 = Math.round(this._p2.x * scope.horizontalPixelRatio);
      var y2 = Math.round(this._p2.y * scope.verticalPixelRatio);
      ctx.lineWidth = this._options.width;
      ctx.strokeStyle = this._options.lineColor;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      if (this._options.showLabels) {
        this._drawTextLabel(scope, this._text1, x1, y1, true);
        this._drawTextLabel(scope, this._text2, x2, y2, false);
      }
    });
  }

  _drawTextLabel(scope, text, x, y, left) {
    var ctx = scope.context;
    ctx.font = "12px inherit";
    ctx.beginPath();
    var offset = 5 * scope.horizontalPixelRatio;
    var width = ctx.measureText(text).width;
    var leftAdjust = left ? width + offset * 4 : 0;
    ctx.fillStyle = this._options.labelBackgroundColor;
    ctx.roundRect(x + offset - leftAdjust, y - 18, width + offset * 2, 18 + offset, 4);
    ctx.fill();
    ctx.beginPath();
    ctx.fillStyle = this._options.labelTextColor;
    ctx.fillText(text, x + offset * 2 - leftAdjust, y - 4);
  }
}

class TrendLinePaneView {
  constructor(source) {
    this._source = source;
    this._p1 = { x: null, y: null };
    this._p2 = { x: null, y: null };
  }

  update() {
    var series = this._source._series;
    var y1 = series.priceToCoordinate(this._source._p1.price);
    var y2 = series.priceToCoordinate(this._source._p2.price);
    var timeScale = this._source._chart.timeScale();
    var x1 = timeScale.timeToCoordinate(this._source._p1.time);
    var x2 = timeScale.timeToCoordinate(this._source._p2.time);
    this._p1 = { x: x1, y: y1 };
    this._p2 = { x: x2, y: y2 };
  }

  renderer() {
    return new TrendLinePaneRenderer(
      this._p1, this._p2,
      this._source._p1.price.toFixed(1),
      this._source._p2.price.toFixed(1),
      this._source._options
    );
  }
}

var TREND_LINE_DEFAULTS = {
  lineColor: "rgb(0, 0, 0)",
  width: 2,
  showLabels: true,
  labelBackgroundColor: "rgba(255, 255, 255, 0.85)",
  labelTextColor: "rgb(0, 0, 0)",
};

class TrendLine {
  constructor(chart, series, p1, p2, options) {
    this._chart = chart;
    this._series = series;
    this._p1 = p1;
    this._p2 = p2;
    this._minPrice = Math.min(p1.price, p2.price);
    this._maxPrice = Math.max(p1.price, p2.price);
    this._options = Object.assign({}, TREND_LINE_DEFAULTS, options);
    this._paneViews = [new TrendLinePaneView(this)];
  }

  autoscaleInfo(startTimePoint, endTimePoint) {
    var p1Index = this._pointIndex(this._p1);
    var p2Index = this._pointIndex(this._p2);
    if (p1Index === null || p2Index === null) return null;
    if (endTimePoint < p1Index || startTimePoint > p2Index) return null;
    return { priceRange: { minValue: this._minPrice, maxValue: this._maxPrice } };
  }

  updateAllViews() {
    this._paneViews.forEach(function (pw) { pw.update(); });
  }

  paneViews() {
    return this._paneViews;
  }

  _pointIndex(p) {
    var coordinate = this._chart.timeScale().timeToCoordinate(p.time);
    if (coordinate === null) return null;
    return this._chart.timeScale().coordinateToLogical(coordinate);
  }
}

window.AlgorixDrawingTools = {
  trendline: {
    pointsNeeded: 2,
    kind: "primitive",
    make: function (chart, series, points, options) {
      return new TrendLine(chart, series, points[0], points[1], options);
    },
    label: function (points) {
      return "Trendline " + points[0].time + " → " + points[1].time;
    },
  },
  breakout: {
    pointsNeeded: 1,
    kind: "marker",
    shape: "arrowUp",
    position: "belowBar",
    colorToken: "--good",
    label: function (points) { return "Breakout marker (" + points[0].time + ")"; },
  },
  dip: {
    pointsNeeded: 1,
    kind: "marker",
    shape: "arrowDown",
    position: "aboveBar",
    colorToken: "--warn",
    label: function (points) { return "Dip marker (" + points[0].time + ")"; },
  },
};
