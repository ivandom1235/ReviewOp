import React from "react";

export default class AppErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, errorMessage: "" };
  }

  static getDerivedStateFromError(error) {
    return {
      hasError: true,
      errorMessage: error instanceof Error ? error.message : "Unexpected UI error",
    };
  }

  componentDidCatch(error, info) {
    console.error("AppErrorBoundary caught a render error", error, info);
  }

  handleReset = () => {
    this.setState({ hasError: false, errorMessage: "" });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="grid min-h-screen place-items-center bg-[#f1f5f9] px-6 text-slate-800">
          <div className="w-full max-w-xl rounded-3xl border border-slate-200 bg-white p-8 shadow-sm">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-red-500">Interface Error</p>
            <h1 className="mt-2 text-2xl font-semibold">The page hit a render error.</h1>
            <p className="mt-3 text-sm text-slate-600">
              {this.state.errorMessage || "A component crashed while rendering the latest result."}
            </p>
            <button
              type="button"
              onClick={this.handleReset}
              className="mt-5 rounded-xl bg-slate-900 px-4 py-2.5 text-sm font-medium text-white hover:bg-slate-800"
            >
              Retry UI
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
