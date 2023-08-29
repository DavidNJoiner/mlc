#include "table_cmd.h"

int sort_column;
Table table;

// Console test

char consoleBuffer[MAX_LINES][LINE_LENGTH];
int consoleLineCount = 0;

void addToConsole(const char *line)
{
    if (consoleLineCount < MAX_LINES)
    {
        strncpy(consoleBuffer[consoleLineCount], line, LINE_LENGTH - 1);
        consoleBuffer[consoleLineCount][LINE_LENGTH - 1] = '\0'; // Ensure null-termination
        consoleLineCount++;
    }
    else
    {
        // Shift lines up
        for (int i = 1; i < MAX_LINES; i++)
        {
            strncpy(consoleBuffer[i - 1], consoleBuffer[i], LINE_LENGTH);
        }
        strncpy(consoleBuffer[MAX_LINES - 1], line, LINE_LENGTH - 1);
        consoleBuffer[MAX_LINES - 1][LINE_LENGTH - 1] = '\0'; // Ensure null-termination
    }
}

void renderConsoleArea(int width, int height)
{
    int consoleHeight = LINE_HEIGHT * MAX_LINES;

    // Render dark background for console area
    glColor3f(0.1, 0.1, 0.1);
    glBegin(GL_QUADS);
    glVertex2f(0, 0);
    glVertex2f(width, 0);
    glVertex2f(width, consoleHeight);
    glVertex2f(0, consoleHeight);
    glEnd();

    // Render each line from the buffer
    int y = consoleHeight - LINE_HEIGHT;
    for (int i = 0; i < consoleLineCount; i++)
    {
        drawString(10, y, consoleBuffer[i], 37); // Assuming white text
        y -= LINE_HEIGHT;
    }
}

// Visual test

// Initialize the OpenGL Window
void initOpenGL(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(1000, 800); // Adjust this to your preferred window size
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Memory Table Display");

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 1000, 0, 800); // Adjust this to your preferred window size
    glMatrixMode(GL_MODELVIEW);

    glClearColor(0.25, 0.25, 0.25, 1.0);

    glutDisplayFunc(gl_display_table);
    glutMainLoop();
}

void setOpenGLColor(int col)
{
    switch (col)
    {
    case 37: // light gray/white
        glColor3f(1.0, 1.0, 1.0);
        break;
    case 32: // green
        glColor3f(0.0, 1.0, 0.0);
        break;
    case 31: // red
        glColor3f(1.0, 0.0, 0.0);
        break;
    default: // default to white
        glColor3f(1.0, 1.0, 1.0);
        break;
    }
}

void drawCell(float x, float y, float width, float height)
{
    glColor3f(0.25, 0.25, 0.25); // Dark grey background
    glRectf(x, y, x + width, y - height);

    glColor3f(0.5, 0.5, 0.5); // White border
    glBegin(GL_LINE_LOOP);
    glVertex2f(x, y);
    glVertex2f(x + width, y);
    glVertex2f(x + width, y - height);
    glVertex2f(x, y - height);
    glEnd();
}

void drawString(float x, float y, const char *string, int col)
{
    setOpenGLColor(col);
    glRasterPos2f(x, y);
    for (const char *c = string; *c != '\0'; c++)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c); // Changed font
    }
}

// Use this function when drawing table headers
void drawHeader(float x, float y, const char *string)
{
    drawString(x, y, string, 37); // 37 for light gray/white
}

// Use this function when drawing values with green or red color
void drawValue(float x, float y, const char *string, float value, int columnIndex)
{
    if (columnIndex == 2 && value != 0.00)
    {
        drawString(x, y, string, 32); // 32 for green
    }
    else if (columnIndex == 3 && value != 0.00)
    {
        drawString(x, y, string, 31); // 31 for red
    }
    else
    {
        drawString(x, y, string, 0); // 0 for default (black)
    }
}

void gl_display_table()
{
    glClear(GL_COLOR_BUFFER_BIT);

    int startY = 550;
    int stepY = 20;
    int cellHeight = 20;                              // same as stepY for simplicity
    int colWidth[6] = {140, 100, 150, 150, 150, 150}; // Widths for each column
    int colX[6] = {10, 150, 250, 400, 550, 700};

    // Draw cells
    for (int i = 0; i < table.num_entries; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            drawCell(colX[j], startY - i * stepY, colWidth[j], cellHeight);
        }
    }

    for (int i = 0; i < table.num_entries; i++)
    {

        drawHeader(colX[0], startY - i * stepY, table.entries[i].name);

        char id[10];
        sprintf(id, "%d", table.entries[i].ID);
        drawHeader(colX[1], startY - i * stepY, id);

        drawHeader(colX[2], startY - i * stepY, table.entries[i].time);

        for (int j = 0; j < table.num_columns - 3; j++)
        {
            char value[10];
            sprintf(value, "%.2f", table.entries[i].values[j + 2]);
            drawValue(colX[3 + j], startY - i * stepY, value, table.entries[i].values[j + 2], j + 2);
        }
    }

    glFlush();
}

// end visual test------------------------------------------------------------------------------------------------------------------------------

void init_table()
{
    table.num_entries = 0;
    table.num_columns = 3; // default columns: Refs, ID, and Time
    strcpy(table.headers[0], "Refs");
    strcpy(table.headers[1], "ID");
    strcpy(table.headers[2], "Time");
    add_column("alloc");
    add_column("dealloc");
    add_column("pool used"); // Add the column here only once
}

void add_column(const char *header)
{
    if (table.num_columns >= MAX_COLUMNS)
    {
        printf("Maximum columns reached!\n");
        return;
    }
    strcpy(table.headers[table.num_columns], header);
    table.num_columns++;
}

void add_entry(char *name, int num_values, ...)
{
    if (table.num_entries >= MAX_ENTRIES)
    {
        printf("Table is full!\n");
        return;
    }
    if (num_values != table.num_columns - 4) // -4 because of Refs, ID, Time columns and ....
    {
        printf("Mismatch in number of values and columns!\n");
        return;
    }
    strcpy(table.entries[table.num_entries].name, name);
    table.entries[table.num_entries].ID = table.num_entries;
    strcpy(table.entries[table.num_entries].time, get_time());

    // Initialize the values array to 0
    for (int i = 0; i < table.num_columns; i++)
    {
        table.entries[table.num_entries].values[i] = 0.0;
    }

    va_list args;
    va_start(args, num_values);
    for (int i = 0; i < num_values; i++)
    {
        table.entries[table.num_entries].values[i + 2] = va_arg(args, double); // Start from index 2
    }
    va_end(args);

    table.entries[table.num_entries].values[table.num_columns - 2] = (double)(get_total_bytes_allocated());

    table.num_entries++;
}

void display_table()
{
    // Adjust these widths as needed
    int nameWidth = 15;
    int idWidth = 5;
    int timeWidth = 10;
    int valueWidth = 10;

    // Print headers in blue
    printf("\033[37m");
    printf("\n%-*s %-*s %-*s", nameWidth, table.headers[0], idWidth, table.headers[1], timeWidth, table.headers[2]);
    for (int i = 3; i < table.num_columns; i++)
    {
        printf("%-*s", valueWidth, table.headers[i]);
    }
    printf("\033[0m\n"); // Reset color

    for (int i = 0; i < table.num_entries; i++)
    {
        printf("%-*s %-*d %-*s", nameWidth, table.entries[i].name, idWidth, table.entries[i].ID, timeWidth, table.entries[i].time);

        for (int j = 2; j < table.num_columns - 1; j++)
        {
            if (j == 2 && table.entries[i].values[j] != 0.00) // Assuming index 2 is 'alloc'
            {
                printf("\033[32m"); // Set text color to green
            }
            else if (j == 3 && table.entries[i].values[j] != 0.00) // Assuming index 3 is 'dealloc'
            {
                printf("\033[31m"); // Set text color to red
            }

            printf("%-*.*f", valueWidth, 2, table.entries[i].values[j]);
            printf("\033[0m"); // Reset color after printing the value
        }
        printf("\n");
    }
    printf("\n");
}

void display_element_by_ID(int ID)
{
    for (int i = 0; i < table.num_entries; i++)
    {
        if (table.entries[i].ID == ID)
        {
            printf("%-10s %-10d", table.entries[i].name, table.entries[i].ID);
            for (int j = 0; j < table.num_columns - 2; j++)
            {
                printf("%-10.2f", table.entries[i].values[j]);
            }
            printf("\n");
            return;
        }
    }
    printf("ID not found!\n");
}

int compare_desc(const void *a, const void *b)
{
    float val1 = ((Entry *)a)->values[sort_column];
    float val2 = ((Entry *)b)->values[sort_column];
    return (val2 > val1) - (val2 < val1);
}

void display_sort_down(int column)
{
    Entry sorted_entries[MAX_ENTRIES];
    memcpy(sorted_entries, table.entries, sizeof(Entry) * MAX_ENTRIES);
    sort_column = column; // Set the global variable
    qsort(sorted_entries, table.num_entries, sizeof(Entry), compare_desc);
    printf("%-10s %-10s", table.headers[0], table.headers[1]);
    for (int i = 2; i < table.num_columns; i++)
    {
        printf("%-10s", table.headers[i]);
    }
    printf("\n");
    for (int i = 0; i < table.num_entries; i++)
    {
        printf("%-10s %-10d", sorted_entries[i].name, sorted_entries[i].ID);
        for (int j = 0; j < table.num_columns - 2; j++)
        {
            printf("%-10.2f", sorted_entries[i].values[j]);
        }
        printf("\n");
    }
}

void free_table()
{
    table.num_entries = 0;
}
