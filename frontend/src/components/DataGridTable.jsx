import { DataGrid } from "@mui/x-data-grid";
import { ThemeProvider, createTheme } from "@mui/material/styles";

export default function DataGridTable({ columns, rows, isDark, height = 360, getRowId }) {
  const theme = createTheme({
    palette: {
      mode: isDark ? "dark" : "light",
      background: {
        default: isDark ? "#020617" : "#f8fafc",
        paper: isDark ? "#0b1220" : "#ffffff",
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
            borderBottom: isDark ? "1px solid #1e293b" : "1px solid #e2e8f0",
            backgroundColor: isDark ? "#0f172a" : "#f1f5f9",
          },
          row: {
            borderBottom: isDark ? "1px solid #1e293b" : "1px solid #e2e8f0",
          },
        },
      },
    },
  });

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
