#include <windows.h>
#include <string>
#include <vector>
#include <thread>
#include <sstream>
#include <algorithm>

#pragma comment(lib, "User32.lib")
#pragma comment(lib, "Gdi32.lib")

// ---------------------------------------------
// Konfiguracja programów
// ---------------------------------------------
struct ProgramEntry {
    std::string name;
    std::string path;
    std::vector<std::string> args;   // format: --KEY=value
};

std::vector<ProgramEntry> programs = {
    {
        "CUDA",
        "diffusion_cuda.exe",
        { "--w=1000", "--h=1000", "--steps=50000", "--tile=32", "--save_every=500", "--repeat=10" }
    },
    {
        "Sequential",
        "diffusion_seq.exe",
        { "--w=100", "--h=100", "--steps=10000", "--save_every=1000", "--repeat=1" }
    },
    {
        "Inny program",
        "other_program.exe",
        { "--param1=10", "--param2=20" }
    }
};

// ---------------------------------------------
// Kontrolki GUI
// ---------------------------------------------
HWND hComboProgram;
HWND hButtonStart;
HWND hEditLog;

std::vector<HWND> argLabels;
std::vector<HWND> argEdits;

// ---------------------------------------------
// Log
// ---------------------------------------------
void AppendLog(const std::string& msg) {
    int len = GetWindowTextLengthA(hEditLog);
    SendMessageA(hEditLog, EM_SETSEL, len, len);
    SendMessageA(hEditLog, EM_REPLACESEL, TRUE, (LPARAM)msg.c_str());
}

// ---------------------------------------------
// Uruchamianie procesów
// ---------------------------------------------
void RunProcess(const std::string& exe, const std::string& args) {
    AppendLog("Uruchamianie: " + exe + " " + args + "\r\n");

    SECURITY_ATTRIBUTES sa{ sizeof(SECURITY_ATTRIBUTES), NULL, TRUE };
    HANDLE hRead, hWrite;

    CreatePipe(&hRead, &hWrite, &sa, 0);
    SetHandleInformation(hRead, HANDLE_FLAG_INHERIT, 0);

    PROCESS_INFORMATION pi{};
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.hStdOutput = hWrite;
    si.hStdError = hWrite;
    si.dwFlags = STARTF_USESTDHANDLES;

    std::string cmd = exe + " " + args;

    if (!CreateProcessA(
        NULL, (LPSTR)cmd.c_str(),
        NULL, NULL, TRUE,
        CREATE_NO_WINDOW,
        NULL, NULL,
        &si, &pi))
    {
        AppendLog("B£¥D: nie mo¿na uruchomiæ programu.\r\n");
        return;
    }

    CloseHandle(hWrite);

    char buffer[256];
    DWORD bytesRead;

    while (ReadFile(hRead, buffer, 255, &bytesRead, NULL) && bytesRead > 0) {
        buffer[bytesRead] = 0;
        AppendLog(buffer);
    }

    WaitForSingleObject(pi.hProcess, INFINITE);

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(hRead);

    AppendLog("Zakoñczono.\r\n");
}

// ---------------------------------------------
// Funkcja pomocnicza do niszczenia kontrolek
// ---------------------------------------------
void DestroyAllArgControls() {
    // Niszczenie wszystkich istniej¹cych kontrolek
    for (HWND h : argLabels) {
        if (h && IsWindow(h)) {
            DestroyWindow(h);
        }
    }

    for (HWND h : argEdits) {
        if (h && IsWindow(h)) {
            DestroyWindow(h);
        }
    }

    // Czyszczenie wektorów
    argLabels.clear();
    argEdits.clear();

    // Wymuszenie odœwie¿enia okna
    InvalidateRect(hComboProgram, NULL, TRUE);
    UpdateWindow(hComboProgram);
}

// ---------------------------------------------
// Tworzenie pól argumentów
// ---------------------------------------------
void CreateArgControls(HWND hwnd, int programIndex) {
    // Usuñ wszystkie istniej¹ce kontrolki argumentów
    DestroyAllArgControls();

    // SprawdŸ, czy indeks jest poprawny
    if (programIndex < 0 || programIndex >= (int)programs.size()) {
        return;
    }

    // ---- tworzenie nowych kontrolek ----
    int y = 60;
    ProgramEntry& program = programs[programIndex];

    for (size_t i = 0; i < program.args.size(); i++) {
        const std::string& arg = program.args[i];
        size_t pos = arg.find('=');

        if (pos == std::string::npos) {
            continue; // Pomijamy nieprawid³owe argumenty
        }

        std::string name = arg.substr(2, pos - 2);
        std::string value = arg.substr(pos + 1);

        // Tworzenie etykiety
        HWND lbl = CreateWindowA("STATIC", name.c_str(),
            WS_CHILD | WS_VISIBLE | SS_RIGHT,
            20, y, 120, 20,
            hwnd, NULL, NULL, NULL);

        // Tworzenie pola edycyjnego
        HWND edt = CreateWindowA("EDIT", value.c_str(),
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
            150, y, 160, 22,
            hwnd, NULL, NULL, NULL);

        // Dodawanie uchwytów do wektorów
        if (lbl) argLabels.push_back(lbl);
        if (edt) argEdits.push_back(edt);

        y += 28;
    }

    // Przesuñ log, ¿eby nie zas³ania³ nowych kontrolek
    if (hEditLog && IsWindow(hEditLog)) {
        SetWindowPos(hEditLog, NULL, 20, y + 20, 450, 200, SWP_NOZORDER);
    }

    // Wymuszenie odœwie¿enia okna
    InvalidateRect(hwnd, NULL, TRUE);
    UpdateWindow(hwnd);
}

// ---------------------------------------------
// Procedura okna
// ---------------------------------------------
LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {

    switch (msg) {

    case WM_CREATE:
        hComboProgram = CreateWindowA("COMBOBOX", "",
            WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_TABSTOP,
            20, 20, 300, 200,
            hwnd, NULL, NULL, NULL);

        hButtonStart = CreateWindowA("BUTTON", "Start",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON | WS_TABSTOP,
            350, 20, 120, 30,
            hwnd, (HMENU)1, NULL, NULL);

        // Pocz¹tkowa pozycja dla logu (zostanie zmieniona w CreateArgControls)
        hEditLog = CreateWindowA("EDIT", "",
            WS_CHILD | WS_VISIBLE | WS_BORDER |
            ES_MULTILINE | ES_AUTOVSCROLL | ES_WANTRETURN | WS_VSCROLL | ES_READONLY,
            20, 250, 450, 200,
            hwnd, NULL, NULL, NULL);

        // Dodanie programów do comboboxa
        for (const auto& p : programs) {
            SendMessageA(hComboProgram, CB_ADDSTRING, 0, (LPARAM)p.name.c_str());
        }

        SendMessageA(hComboProgram, CB_SETCURSEL, 0, 0);

        // Utworzenie pocz¹tkowych kontrolek
        CreateArgControls(hwnd, 0);
        break;

    case WM_COMMAND:
        // zmiana programu
        if (HIWORD(wParam) == CBN_SELCHANGE && (HWND)lParam == hComboProgram) {
            int idx = SendMessageA(hComboProgram, CB_GETCURSEL, 0, 0);
            if (idx != CB_ERR) {
                CreateArgControls(hwnd, idx);
            }
        }

        // przycisk START
        if (LOWORD(wParam) == 1) {
            int pidx = SendMessageA(hComboProgram, CB_GETCURSEL, 0, 0);

            if (pidx == CB_ERR || pidx < 0 || pidx >= (int)programs.size()) {
                AppendLog("B£¥D: Nie wybrano programu!\r\n");
                break;
            }

            std::string args = "";
            ProgramEntry& program = programs[pidx];

            // SprawdŸ czy liczba kontrolek odpowiada liczbie argumentów
            if (argEdits.size() != program.args.size()) {
                AppendLog("B£¥D: Nieprawid³owa liczba argumentów!\r\n");
                break;
            }

            for (size_t i = 0; i < argEdits.size(); i++) {
                char buf[256];
                GetWindowTextA(argEdits[i], buf, sizeof(buf));

                std::string key = program.args[i];
                size_t pos = key.find('=');

                if (pos != std::string::npos) {
                    key = key.substr(0, pos);
                }

                args += key + " " + std::string(buf) + " ";
            }

            std::thread t(RunProcess, program.path, args);
            t.detach();
        }
        break;

    case WM_SIZE: {
        // Dopasowanie rozmiaru logu przy zmianie rozmiaru okna
        int width = LOWORD(lParam);
        int height = HIWORD(lParam);

        if (hEditLog && IsWindow(hEditLog)) {
            // ZnajdŸ najni¿sz¹ kontrolkê argumentów
            int maxY = 60 + (int)argEdits.size() * 28 + 40;

            SetWindowPos(hEditLog, NULL,
                20, maxY,
                width - 40, height - maxY - 20,
                SWP_NOZORDER);
        }
        break;
    }

    case WM_DESTROY:
        // Zniszcz wszystkie kontrolki przed zamkniêciem
        DestroyAllArgControls();
        PostQuitMessage(0);
        break;

    case WM_CLOSE:
        DestroyWindow(hwnd);
        break;
    }

    return DefWindowProc(hwnd, msg, wParam, lParam);
}

// ---------------------------------------------
// WinMain
// ---------------------------------------------
int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int) {

    WNDCLASSA wc{};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = "ColorDiffusionGUIClass";
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);

    RegisterClassA(&wc);

    HWND hwnd = CreateWindowA(
        "ColorDiffusionGUIClass", "Color Diffusion Program Runner",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, 520, 520,
        NULL, NULL, hInst, NULL);

    if (!hwnd) {
        return 0;
    }

    MSG msg;
    while (GetMessageA(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
    return (int)msg.wParam;
}