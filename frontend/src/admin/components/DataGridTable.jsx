import { DataGrid } from "@mui/x-data-grid";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import { memo } from "react";

const lightTheme = createTheme({
  palette: {
    mode: "light",
    background: {
      default: "#f8fafc",
      paper: "#ffffff",
    },
  },
  components: {
    MuiDataGrid: {
      styleOverrides: {
        root: {
          border: 0,
          fontSize: 13,
        },
        columnHeaders: {
          borderBottom: "1px solid #e2e8f0",
          backgroundColor: "#f1f5f9",
        },
        row: {
          borderBottom: "1px solid #e2e8f0",
        },
      },
    },
  },
});

const darkTheme = createTheme({
  palette: {
    mode: "dark",
    background: {
      default: "#020617",
      paper: "#0b1220",
    },
  },
  components: {
    MuiDataGrid: {
      styleOverrides: {
        root: {
          border: 0,
          fontSize: 13,
        },
        columnHeaders: {
          borderBottom: "1px solid #1e293b",
          backgroundColor: "#0f172a",
        },
        row: {
          borderBottom: "1px solid #1e293b",
        },
      },
    },
  },
});

function DataGridTable({ columns, rows, isDark, height = 360, getRowId }) {
  const darkMode = typeof isDark === "boolean" ? isDark : document.documentElement.classList.contains("dark");
  const theme = darkMode ? darkTheme : lightTheme;

  return (
    <ThemeProvider theme={theme}>
      <div style={{ height, width: "100%" }}>
        <DataGrid
          rows={rows}
          columns={columns}
          getRowId={getRowId}
          disableRowSelectionOnClick
          initialState={{
            pagination: { paginationModel: { page: 0, pageSize: 8 } },
          }}
          pageSizeOptions={[8, 15, 25]}
          density="compact"
        />
      </div>
    </ThemeProvider>
  );
}

export default memo(DataGridTable);
