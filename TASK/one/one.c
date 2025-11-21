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
} CartItem;

Product PRODUCTS[] = {
    {1, "Eraser", 12.0, 0},
    {2, "Sharpener", 18.0, 0},
    {3, "Wheat", 75.0, 0},
    {4, "Tea", 220.0, 1},
    {5, "Pepper", 35.0, 0},
    {6, "Honey", 95.0, 0},
    {7, "Curd", 55.0, 0},
    {8, "Cheese", 150.0, 1},
    {9, "Butter", 110.0, 1},
    {10, "Sketchbook", 65.0, 0}};

int PRODUCT_COUNT = sizeof(PRODUCTS) / sizeof(PRODUCTS[0]);

int main()
{
    CartItem *cart = malloc(2 * sizeof(CartItem));
    if (!cart)
    {
        printf("Memory allocation failed\n");
        return 1;
    }

    int capacity = 2;
    int item_count = 0;
    int invoice_no = 1;

    FILE *invFile = fopen("invoiceNumber.txt", "r");
    if (invFile)
    {
        fscanf(invFile, "%d", &invoice_no);
        fclose(invFile);
    }

    printf("┌────┬────────────┬────────┬─────────┐\n");
    printf("│ ID │ Name       │ Price  │ Taxable │\n");
    printf("├────┼────────────┼────────┼─────────┤\n");
    for (int i = 0; i < PRODUCT_COUNT; i++)
    {
        printf("│ %2d │ %-10s │ %6.2f │    %s   │\n",
               PRODUCTS[i].id,
               PRODUCTS[i].name,
               PRODUCTS[i].price,
               PRODUCTS[i].taxable ? "Yes" : "No");
    }
    printf("└────┴────────────┴────────┴─────────┘\n");

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
                while (getchar() != '\n')
                    ;
                printf("Invalid input. Please enter a valid Product ID.\n");
                continue;
            }

            Product *p = NULL;
            for (int i = 0; i < PRODUCT_COUNT; i++)
            {
                if (PRODUCTS[i].id == pid)
                {
                    p = &PRODUCTS[i];
                    break;
                }
            }
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
                while (getchar() != '\n')
                    ;
                printf("Invalid Quantity. Please enter a positive number.\n");
                continue;
            }
            break;
        }
        // halnu agadi check garxa
        if (item_count == capacity)
        {
            int new_capacity = capacity + 2;
            CartItem *newMemory = realloc(cart, new_capacity * sizeof(CartItem));
            if (!newMemory)
            {
                printf("Memory allocation failed\n");
                free(cart);
                return 1;
            }
            cart = newMemory;
            capacity = new_capacity;
        }

        Product *p = NULL;
        for (int i = 0; i < PRODUCT_COUNT; i++)
        {
            if (PRODUCTS[i].id == pid)
            {
                p = &PRODUCTS[i];
                break;
            }
        }
        cart[item_count].product = p;
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