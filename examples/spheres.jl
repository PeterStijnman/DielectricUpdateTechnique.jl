
p1 = plot([log.(abs.(Einc)) log.(abs.(Etot)) log.(abs.(Etot_update))],ylim=[-17,1]);
p2 = plot([log.(abs.(Etot-Einc)) log.(abs.(Etot-Etot_update))],ylim=[-17,1]);
plot(p1,p2,layout=(2,1),size=(700,700))|>electrondisplay;

Einc_x = Einc[1:13*12*12] |> x -> reshape(x,13,12,12);
Etot_x = Etot[1:13*12*12] |> x -> reshape(x,13,12,12);
Etot_update_x = Etot_update[1:13*12*12] |> x -> reshape(x,13,12,12);

slice = 6
h1 = heatmap(log.(abs.(Einc_x[:,:,slice])));
h2 = heatmap(log.(abs.(Etot_x[:,:,slice])));
h3 = heatmap(log.(abs.(Etot_update_x[:,:,slice])));
plot(h1,h2,h3,size=(750,750))|> electrondisplay;