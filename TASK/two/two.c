#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define NUM_THREADS 3

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
typedef struct
{
    int start;
    int end;
    CartItem *cart;
    double discount_percent;
    double local_subtotal;
    double local_discount;
    double local_taxable;
} WorkerArgs;
typedef struct
{
    double *subtotal;
    double *discount_amount;
    double *taxable_amount;
} ResultRef;
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
const double TAX_RATE = 0.13;
// this function returns a pointer to the product with the given id
// if the product is not found, it returns NULL
Product *find_product_by_id(int id)
{
    for (int i = 0; i < PRODUCT_COUNT; i++)
    {
        if (PRODUCTS[i].id == id)
            return &PRODUCTS[i];
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
               PRODUCTS[i].id,
               PRODUCTS[i].name,
               PRODUCTS[i].price,
               PRODUCTS[i].taxable ? "Yes" : "No");
    }
    printf("└────┴────────────┴────────┴─────────┘\n");
}
void *worker_thread(void *arg)
{
    WorkerArgs *w = (WorkerArgs *)arg;
    double sub = 0.0, disc = 0.0, tax = 0.0;
    for (int i = w->start; i < w->end; i++)
    {
        Product *p = w->cart[i].product;
        int q = w->cart[i].qty;
        double line_total = p->price * q;
        sub += line_total;
        disc += (line_total * w->discount_percent) / 100.0;
        if (p->taxable)
            tax += line_total;
    }
    w->local_subtotal = sub;
    w->local_discount = disc;
    w->local_taxable = tax;
    return NULL;
}
int main()
{
    CartItem *cart = malloc(2 * sizeof(CartItem));
    int capacity = 2;
    int item_count = 0;
    int invoice_no = 1;
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
                while (getchar() != '\n')
                    ;
                printf("Invalid Quantity. Please enter a positive number.\n");
                continue;
            }
            break;
        }
        if (item_count == capacity)
        {
            int new_capacity = capacity * 2;
            CartItem *newMemory = realloc(cart, new_capacity * sizeof(CartItem));
            if (!newMemory)
            {
                printf("Memory allocation failed\n");
                free(cart);
                return 1;
            }
            // Yo kina chaiyo? ==> if realloc ley meomery expand garna sakena vane, it will create a new momory and store the old data. so to avoid the conflit when thsi happens, why not copy the reallocked address to cart again.
            cart = newMemory;
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
    printf("Enter discount percentage (0-100): ");
    double discount_percent;
    scanf("%lf", &discount_percent);
    if (discount_percent < 0.0)
        discount_percent = 0.0;
    if (discount_percent > 100.0)
        discount_percent = 100.0;
    double subtotal = 0.0, discount_amount = 0.0, taxable_amount = 0.0;

    pthread_t threads[NUM_THREADS];
    WorkerArgs args[NUM_THREADS];
    int chunk = item_count / NUM_THREADS;
    int rem = item_count % NUM_THREADS;
    int start = 0;
    for (int t = 0; t < NUM_THREADS; t++)
    {
        int end = start + chunk + (t < rem ? 1 : 0);
        args[t].start = start;
        args[t].end = end;
        args[t].cart = cart;
        args[t].discount_percent = discount_percent;
        args[t].local_subtotal = 0.0;
        args[t].local_discount = 0.0;
        args[t].local_taxable = 0.0;
        pthread_create(&threads[t], NULL, worker_thread, &args[t]);
        start = end;
    }
    for (int t = 0; t < NUM_THREADS; t++)
    {
        pthread_join(threads[t], NULL);
        subtotal += args[t].local_subtotal;
        discount_amount += args[t].local_discount;
        taxable_amount += args[t].local_taxable;
    }

    double discount_on_taxable = (taxable_amount * discount_percent) / 100.0;
    double taxable_after_discount = taxable_amount - discount_on_taxable;
    double final_tax = taxable_after_discount * TAX_RATE;
    double grand_total = subtotal - discount_amount + final_tax;
    printf("\n┌──────────────────────────────────────────────┐\n");
    printf("│              INVOICE  #%04d                   │\n", invoice_no);
    printf("├────┬──────────────┬──────────┬──────┬────────┬──────────┐\n");
    printf("│ SN │ Product      │ Quantity │ Rate │ Total  │ Taxable │\n");
    printf("├────┼──────────────┼──────────┼──────┼────────┼──────────┤\n");
    for (int i = 0; i < item_count; i++)
    {
        Product *p = cart[i].product;
        int q = cart[i].qty;
        double total = p->price * q;
        printf("│ %2d │ %-12s │ %8d │ %4.2f │ %6.2f │ %d │\n",
               i + 1, p->name, q, p->price, total, p->taxable);
    }

    printf("├────┴──────────────┴──────────┴──────┴────────┤\n");
    printf("│                              Subtotal: %6.2f │\n", subtotal);
    printf("│                            Discount: %6.2f │\n", discount_amount);
    printf("│                      Tax (13.0%%): %6.2f │\n", final_tax);
    printf("│                        Grand Total: %6.2f │\n", grand_total);
    printf("└──────────────────────────────────────────────────┘\n");

    invFile = fopen("invoiceNumber.txt", "w");
    if (invFile)
    {
        fprintf(invFile, "%d\n", invoice_no + 1);
        fclose(invFile);
    }

    free(cart);
    return 0;
}