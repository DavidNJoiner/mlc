#ifndef TABLE_H
#define TABLE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>

#define MAX_ENTRIES 200
#define MAX_COLUMNS 10
#define MAX_NAME_LEN 50
#define MAX_HEADER_LEN 50
#define TIME_LEN 80

typedef struct
{
    char name[MAX_NAME_LEN];
    int ID;
    char time[TIME_LEN];
    float values[MAX_COLUMNS];
} Entry;

typedef struct
{
    Entry entries[MAX_ENTRIES];
    char headers[MAX_COLUMNS][MAX_HEADER_LEN];
    int num_entries;
    int num_columns;
} Table;

extern int sort_column;
extern Table table;

void init_table();
void add_column(const char *header);
void add_entry(char *name, int num_values, ...);
void display_table();
void display_element_by_ID(int ID);
int compare_desc(const void *a, const void *b);
void display_sort_down(int column);
void free_table();
char *get_time();

#endif // TABLE_H