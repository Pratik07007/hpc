#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct
{
    int id;
    char *name;
    double price;
    int taxable;
} Product;

typedef struct
{
    Product *product;
    int qty;
} OrderItem;

Product Products[] = {
    {1, "Pen", 25.0, 0},
    {2, "Pencil", 15.0, 0},
    {3, "Rice", 80.0, 0},
    {4, "Coffee", 250.0, 1},
    {5, "Salt", 30.0, 0},
    {6, "Sugar", 45.0, 0},
    {7, "Milk", 60.0, 0},
    {8, "Paneer", 120.0, 1},
    {9, "Tofu", 90.0, 1},
    {10, "Notebook", 40.0, 0}};

int PRODUCT_COUNT = sizeof(Products) / sizeof(Products[0]);

// this function returns a pointer to the product with the given id
// if the product is not found, it returns NULL
Product *find_product_by_id(int id)

{
    for (int i = 0; i < PRODUCT_COUNT; i++)
    {
        if (Products[i].id == id)
            return &Products[i];
    }
    return NULL;
}

void print_products_list()
{
    printf("┌────┬────────────┬────────┬─────────┐\n");
    printf("│ ID │ Name       │ Price  │ Taxable │\n");
    printf("├────┼────────────┼────────┼─────────┤\n");
    for (int i = 0; i < PRODUCT_COUNT; i++)
    {
        printf("│ %2d │ %-10s │ %6.2f │    %s   │\n",
               Products[i].id,
               Products[i].name,
               Products[i].price,
               Products[i].taxable ? "Yes" : "No");
    }
    printf("└────┴────────────┴────────┴─────────┘\n");
}

int main()
{
    OrderItem *cart = malloc(2 * sizeof(OrderItem));
    int capacity = 2;
    int item_count = 0;
    int invoice_no = 1;

    // Read the new invoice number from file
    FILE *invFile = fopen("invoiceNumber.txt", "r");
    if (invFile)
    {
        fscanf(invFile, "%d", &invoice_no);
        fclose(invFile);
    }

    print_products_list();

    while (1)
    {
        int pid;
        int qty;
        char cont;

        while (1)
        {
            printf("\nEnter Product ID: ");
            if (scanf("%d", &pid) != 1)
            {
                // Clear invalid input
                while (getchar() != '\n')
                    ;
                printf("Invalid input. Please enter a valid Product ID.\n");
                continue;
            }

            Product *p = find_product_by_id(pid);
            if (!p)
            {
                printf("Invalid Product ID. Please try again.\n");
                continue;
            }
            break;
        }

        while (1)
        {
            printf("Enter Quantity: ");
            if (scanf("%d", &qty) != 1 || qty <= 0)
            {
                // Clear invalid input
                while (getchar() != '\n')
                    ;
                printf("Invalid Quantity. Please enter a positive number.\n");
                continue;
            }
            break;
        }

        if (item_count == capacity)
        {
            int new_capacity = capacity == 0 ? 2 : capacity * 2;
            OrderItem *temp = realloc(cart, new_capacity * sizeof(OrderItem));
            if (!temp)
            {
                printf("Memory allocation failed\n");
                free(cart);
                return 1;
            }
            cart = temp;
            capacity = new_capacity;
        }

        cart[item_count].product = find_product_by_id(pid);
        cart[item_count].qty = qty;
        item_count++;

        printf("Do you want to place another order? (y/n): ");
        scanf(" %c", &cont);
        if (cont == 'n' || cont == 'N')
            break;
    }

    printf("\n┌──────────────────────────────────────────────┐\n");
    printf("│              INVOICE  #%04d                   │\n", invoice_no);
    printf("├────┬──────────────┬──────────┬──────┬────────┤\n");
    printf("│ SN │ Product      │ Quantity │ Rate │ Total  │\n");
    printf("├────┼──────────────┼──────────┼──────┼────────┤\n");

    double grand_total = 0.0;
    for (int i = 0; i < item_count; i++)
    {
        Product *p = cart[i].product;
        int q = cart[i].qty;
        double total = p->price * q;
        grand_total += total;
        printf("│ %2d │ %-12s │ %8d │ %4.2f │ %6.2f │\n",
               i + 1, p->name, q, p->price, total);
    }

    printf("├────┴──────────────┴──────────┴──────┴────────┤\n");
    printf("│                              Grand Total: %6.2f │\n", grand_total);
    printf("└──────────────────────────────────────────────────┘\n");

    // Write the next invoice number to file
    invFile = fopen("invoiceNumber.txt", "w");
    if (invFile)
    {
        fprintf(invFile, "%d\n", invoice_no + 1);
        fclose(invFile);
    }

    free(cart);
    return 0;
}